(ns raster.compiler.backend.gpu.segop-opencl-test
  "SegRed OpenCL kernel generation — pins #55: aget lowering must NOT be
   ns-sensitive. A parametric-array kernel's reads arrive DEVIRTUALIZED
   ((.invk aget-impl arr i) with :raster.op/original), a typed-array kernel's
   as bare clojure.core/aget — both must lower to the subscript `arr[i]`.
   Before the fix SegRed skipped normalize-array-prims (SegMap didn't) and its
   let*-rewrap dropped the metadata, so the devirtualized shape fell through
   to a broken gpufn_aget helper call (qlinear-k) while the typed shape
   emitted x[i] (decoder-gpu)."
  (:require [clojure.walk :as walk]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.map-read-requirements :as map-reads]
            [raster.compiler.passes.parallel.segmap-capacity-fixture :as capacity-fixture]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.segstencil-body :as segstencil-body]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.backend.gpu.segop-opencl :as sg]
            [raster.compiler.backend.gpu.kernel-body-opencl :as scalar-emitter]
            [raster.gpu.core :as gpu]))

(defn- affine-operation []
  (segop/->SegMap
   903 (segop/make-seg-space 'row 'n) (segop/->SegLevel :thread :virtual)
   '(double (clojure.core/* (long row) (long factor)))
   nil #{} #{'output} #{'factor}
   (segop/->KernelGrid 1 256 0) :double 'output nil))

(deftest canonical-affine-map-preserves-conditional-product-obligations
  ;; Same checked flat-index/mask/Long parameter topology as the captured
  ;; row*nrows, hq*head-dim and row*width bodies; no buffer-capacity inference.
  (let [operation (affine-operation)
        options {:scalar-types {'n :long 'factor :long} :array-types {'output :double}}
        required [{:expression '_n_bound :op :>= :value 0}
                  {:expression 'factor :op :>= :value 0}
                  {:expression (klaunch/product '_n_bound 'factor)
                   :op :<= :value Long/MAX_VALUE}]
        result (try
                 {:artifact (sg/generate-scheduled-segmap-kernel
                              operation :scalar-types (:scalar-types options)
                              :array-types (:array-types options)
                              :target-dialect :opencl-portable)}
                 (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
    (is (nil? (:error result)) "a checked conditional affine schedule must emit portably")
    (when-let [artifact (:artifact result)]
      (is (= required (get-in artifact [:attributes :kernel-body :attributes :required-preconditions])))
      (is (= required (:preconditions artifact)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kart/validate! (assoc artifact :preconditions [])))
          "retained body authority cannot survive dropped obligations"))))

(deftest conditional-affine-index-proof-follows-the-defining-chain
  (let [scheduled (segmap-body/schedule
                   (affine-operation)
                   {:scalar-types {'n :long 'factor :long} :array-types {'output :double}})
        checked (:body scheduled)
        [group lane row] (:indices checked)
        unsafe (kernel-body/->IndexCompute
                'overflowing-origin
                (kernel-body/index-cast
                 (kernel-body/expression :add
                                         (kernel-body/expression :mul (:id group) 256)
                                         (:id lane)) :long :exact))
        alias (assoc row :expression (kernel-body/index-cast 'overflowing-origin :long :exact))
        forged (assoc checked :indices [group lane unsafe alias])
        safe-origin (assoc row :id 'safe-origin)
        safe-alias (assoc row :expression (kernel-body/index-cast 'safe-origin :long :exact))
        valid-alias (assoc checked :indices [group lane safe-origin safe-alias])]
    (is (= checked (kernel-body/validate! checked)))
    (is (= valid-alias (kernel-body/validate! valid-alias))
        "an alias preserves an independently proved widened origin")
    (is (thrown? clojure.lang.ExceptionInfo (kernel-body/validate! forged))
        "an exact alias cannot erase the earlier Int arithmetic overflow boundary")))

(defn- affine-artifact [target]
  (sg/generate-scheduled-segmap-kernel
   (affine-operation) :scalar-types {'n :long 'factor :long}
   :array-types {'output :double} :target-dialect target))

(defn- affine-arguments [artifact n factor pointer]
  (mapv (fn [slot]
          (case (:name slot)
            output pointer
            factor {:type :long :value factor}
            _n_bound {:type :long :value n})) (:abi artifact)))

(defn- affine-graph [artifact]
  (let [{:keys [abi arguments]} (kgraph/public-interface (:abi artifact) (:arguments artifact))]
    (kgraph/make
     {:outputs [(kgraph/buffer 'output :double 'n :device :output)]
      :scalars [(kgraph/->GraphScalar 'factor :long) (kgraph/->GraphScalar 'n :long)]
      :abi abi :arguments arguments
      :nodes [(kgraph/->ScheduledKernel :affine artifact [(kgraph/->ValueUse 'output :write)]
                                       #{'n 'factor} [])]})))

(deftest conditional-affine-contract-is-closed-through-emission-and-artifacts
  (doseq [target [:opencl-portable :opencl-intel :cuda :hip]]
    (let [artifact (affine-artifact target)
          checked (get-in artifact [:attributes :kernel-body])
          scheduled (get-in artifact [:attributes :scheduled-kernel-body])
          conditions (:preconditions artifact)]
      (is (re-find #"\(long(?: long)?\).*map_group.*\*.*\(long(?: long)?\).*256" (:source artifact))
          "target source widens before hardware index multiplication")
      (is (= conditions (:required-preconditions
                         (scalar-emitter/emit-scalar-module
                          "affine_module" checked {:target-dialect target}))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"discards conditional"
                           (scalar-emitter/emit-scalar-kernel
                            "affine_raw" checked {:target-dialect target})))
      (doseq [bad [[] (vec (reverse conditions))
                   (assoc-in conditions [1 :expression] '_n_bound)]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (scheduled-body/validate! (assoc scheduled :preconditions bad))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (kart/validate! (assoc artifact :preconditions bad)))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kart/validate! (update artifact :arguments #(vec (reverse %))))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kernel-body/validate! (update checked :attributes dissoc :required-preconditions))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kernel-body/validate! (assoc checked :operations
                                                (:operations (first (:operations checked)))))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kernel-body/validate! (assoc-in checked [:masks 0 :predicates 0 :op] :gte)))))))

(declare emitted-scan-graph)

(deftest conditional-artifacts-reconcile-every-retained-body-authority
  (let [artifact (affine-artifact :opencl-portable)
        conditional (get-in artifact [:attributes :kernel-body])
        unconditional (assoc conditional :operations [] :attributes {})
        ordinary (sg/generate-scheduled-segmap-kernel
                  (assoc (affine-operation) :lambda '(double factor))
                  :scalar-types {'n :long 'factor :long}
                  :array-types {'output :double} :target-dialect :opencl-portable)]
    (is (= unconditional (kernel-body/validate! unconditional))
        "the contradictory preferred body is independently valid, not malformed setup")
    (is (= artifact (kart/validate! artifact)))
    (is (empty? (:preconditions ordinary)))
    (is (= ordinary (kart/validate! ordinary)))
    (let [scan (get-in (emitted-scan-graph
                       '(raster.par/scan out acc 0.0 i 64 double (+ acc (aget values i))))
                      [:nodes 0 :operation])]
      (is (= "raster.compiler.ir.segop.SegScan"
             (some-> scan :provenance :scheduled-operation class .getName)))
      (is (= scan (kart/validate! scan))
          "ordinary scan keeps its semantic source certificate, not a fabricated body schedule"))
    (doseq [path [[:attributes :kernel-body]
                  [:attributes :scheduled-kernel-body :body]
                  [:provenance :scheduled-operation :body]]]
      (let [copied (-> artifact (assoc :preconditions []) (assoc-in path unconditional))]
        (is (thrown? clojure.lang.ExceptionInfo (kart/validate! copied))
            (str "weaker body must not hide other retained authority: " path))))
    (doseq [path [[:attributes :kernel-body]
                  [:attributes :scheduled-kernel-body]
                  [:provenance :scheduled-operation]]]
      (is (thrown? clojure.lang.ExceptionInfo (kart/validate! (assoc-in artifact path nil)))
          (str "conditional transport requires its complete retained authority: " path)))
    (doseq [path [[:attributes :scheduled-kernel-body]
                  [:provenance :scheduled-operation]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (kart/validate! (update-in artifact path assoc :preconditions [])))
          (str "contradictory schedule conditions cannot choose the weaker certificate: " path))
      (doseq [wrong [(into {} (get-in artifact path)) (affine-operation) {}]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (kart/validate! (assoc-in artifact path wrong)))
            (str "conditional schedule requires its exact record authority: " path))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (kart/validate!
                  (-> artifact
                      (assoc-in [:attributes :kernel-body] nil)
                      (assoc-in [:attributes :scheduled-kernel-body] nil)
                      (assoc-in [:provenance :scheduled-operation] nil))))
        "retained scheduled-body markers cannot survive complete body/certificate removal")))

(deftest conditional-affine-checks-actual-scalars-before-direct-or-graph-contact
  (let [artifact (affine-artifact :opencl-portable)
        graph (affine-graph artifact)
        contacts (atom [])
        sess (atom {:device-id :ocl:0 :session-id :affine-test :closed? false
                    :buffers {} :allocations {} :kernel-graphs {} :events {}})]
    (doseq [[n factor] [[2 4294967296] [1 Long/MAX_VALUE] [0 Long/MAX_VALUE] [7 0]]]
      (is (kcall/validate-preconditions! artifact (affine-arguments artifact n factor nil)))
      (is (= graph (graph-call/preflight! graph
                                        {'n {:type :long :value n}
                                         'factor {:type :long :value factor}}))))
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name] (swap! contacts conj name)
         (throw (ex-info "unexpected native contact" {:reason :unexpected-native-contact})))}
      (fn []
        (doseq [[n factor error] [[-1 1 clojure.lang.ExceptionInfo]
                                  [1 -1 clojure.lang.ExceptionInfo]
                                  [2 Long/MAX_VALUE ArithmeticException]]]
          (let [arguments (affine-arguments artifact n factor :missing-pointer)
                scalars {'n {:type :long :value n} 'factor {:type :long :value factor}}]
            (is (instance? error (try (gpu/bind-kernel-call! sess :same-key artifact arguments)
                                      nil (catch Throwable e e))))
            (is (instance? error (try (gpu/bind-kernel-graph! sess :same-key graph
                                                            {'output :missing-pointer} scalars)
                                      nil (catch Throwable e e))))
            (is (instance? error (try (graph-call/temporary-specs graph scalars)
                                      nil (catch Throwable e e))))))
        (is (empty? @contacts) "allocation, view materialization and registration are never contacted")
        (is (empty? (:kernel-graphs @sess)) "failed rebind does not publish or retire a recording")))))

(deftest portable-map-empty-extent-has-a-masked-valid-launch
  (let [operation (segop/->SegMap
                   902 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(clojure.core/aget input i) nil #{'input} #{'output} #{}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        kernel-body (:kernel-body
                     (segmap-body/lower operation
                                        {:array-types {'input :float 'output :float}
                                         :workgroup-size 32}))
        spec (:launch kernel-body)]
    (doseq [[extent groups] [[0 1] [1 1] [32 1] [33 2]]]
      (is (= [groups] (:group-count (klaunch/realize spec
                                                    (fn [_] extent))))))
    (is (= :map-active (:id (first (:masks kernel-body)))))
    (is (= (kernel-body/predicate :lt 'i
                                 (kernel-body/index-cast '_n_bound :long :exact))
           (first (:predicates (first (:masks kernel-body))))))))

(deftest map-access-proof-declines-negative-and-indirect-coordinates
  (doseq [expression ['(aget a (- i 1))
                      '(aget a (int (aget b j)))]]
    (let [graph (capacity-fixture/inferred-graph expression)
          operation (get-in graph [:nodes 0 :operation])]
      (is (nil? (map-reads/static-read-requirements
                 operation {:dtype :float :array-types {'a :float 'b :float 'C :float}}))))))

(deftest public-outer-product-derives-unknown-input-capacities
  (doseq [[expression expected]
          [['(* (aget a i) (aget b j)) {'a 4 'b 3}]
           ['(+ (aget a i) (aget a j)) {'a 4}]
           ['(let* [k (+ i j)] (aget a k)) {'a 6}]]]
    (let [graph (capacity-fixture/inferred-graph expression)
          emitted (sg/generate-kernel-graph graph)
          artifact (get-in emitted [:nodes 0 :operation])]
      (is (= expected (into {} (map (juxt :id :elements)) (:inputs graph))))
      (is (= :kernel-body (kart/emission-route artifact)))
      (is (= (into {} (map (fn [[id n]] [id [n]])) expected)
             (into {} (comp (filter #(= :input (:kind %))) (map (juxt :id :shape)))
                   (get-in artifact [:attributes :kernel-body :parameters])))))))

(deftest typed-map-preserves-independent-static-graph-capacities
  (doseq [a-capacity [4 8]
          dialect [:opencl-intel :cuda :hip]]
    (let [graph (capacity-fixture/graph a-capacity)
          emitted (sg/generate-kernel-graph graph :target-dialect dialect)
          artifact (get-in emitted [:nodes 0 :operation])
          body (get-in artifact [:attributes :kernel-body])
          shapes (into {} (map (juxt :id :shape)) (:parameters body))
          certificate (get-in artifact [:attributes :scheduled-kernel-body])]
      (is (= :kernel-body (kart/emission-route artifact)))
      (is (= [a-capacity] (shapes 'a)))
      (is (= [3] (shapes 'b)))
      (is (= [12] (shapes 'C)))
      (is (= [a-capacity 3] (mapv :elements (:inputs emitted))))
      (is (= 12 (get-in body [:attributes :extent])))
      (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact))))))

(deftest map-capacity-projection-rejects-drift-without-changing-the-algorithm
  (let [graph (capacity-fixture/graph)
        emit #(get-in (sg/generate-kernel-graph %) [:nodes 0 :operation])
        small (emit graph)
        larger (emit (capacity-fixture/graph 8))
        source #(str/replace (:source %) (:kernel-name %) "map_capacity")
        candidate (get-in small [:attributes :scheduled-kernel-body])
        corrupted (assoc-in candidate [:body :parameters 0 :shape] [12])]
    (is (= (source small) (source larger)))
    (is (= (:arguments small) (:arguments larger)))
    (is (= (:launch small) (:launch larger)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"capacity differs"
                         (segmap-body/validate-static-graph-capacities!
                          corrupted (first (:nodes graph)) graph)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"capacity differs"
                         (segmap-body/validate-static-graph-capacities!
                          (assoc-in candidate [:body :parameters 0 :layout] nil)
                          (first (:nodes graph)) graph)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exact graph node"
                         (segmap-body/validate-static-graph-capacities!
                          candidate (assoc (first (:nodes graph)) :id :foreign-node) graph)))))

(deftest emitted-graph-interface-preserves-in-place-buffer-direction
  (let [operation (segop/->SegMap
                   901 (segop/make-seg-space 'i 'n) (segop/->SegLevel :thread :virtual)
                   '(inc (clojure.core/aget output i)) nil #{'output} #{'output} #{}
                   (segop/->KernelGrid 1 32 0) :float 'output nil)
        scheduled (kgraph/from-segops [operation]
                                      {:inputs #{'output} :outputs #{'output} :dtype :float})
        emitted (sg/generate-kernel-graph
                 scheduled :array-types {'output :float} :scalar-types {'n :int})]
    (is (= ['output 'n] (:arguments emitted)))
    (is (= [:inout :scalar] (mapv :kind (:abi emitted))))
    (let [parameters (get-in emitted [:nodes 0 :operation :attributes :kernel-body :parameters])]
      (is (= 1 (count (filter #(= :inout (:kind %)) parameters))))
      (is (= ['_n_bound] (:shape (first parameters)))
          "unknown graph capacities retain the compatibility shape without new scalar arguments"))
    (is (= :inout (get-in emitted [:inputs 0 :role])))))

(deftest two-phase-reduction-graph-emits-its-explicit-scheduled-dataflow
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double}
                 :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled)
        emitted (sg/generate-kernel-graph
                 graph :array-types {'a :double} :scalar-types {'n :long})
        [phase-one phase-two] (mapv :operation (:nodes emitted))
        source-nodes (:nodes graph)
        certificates (mapv #(get-in % [:provenance :scheduled-operation])
                           [phase-one phase-two])
        partials (first (:inputs (second (mapv :operation (:nodes graph)))))
        result-id (:id (first (:outputs graph)))
        temporary-specs (graph-call/temporary-specs
                         emitted {'n {:type :long :value 1025}})]
    (is (= [(kgraph/scalar 'n :long)] (:scalars graph)))
    (is (= (:scalars graph) (:scalars emitted)))
    (let [slot (->> (map vector (:abi emitted) (:arguments emitted))
                    (some (fn [[slot argument]] (when (= 'n argument) slot))))]
      (is (= :long (:dtype slot))
          "the scheduled graph supplies the public logical dtype")
      (is (= :long (:kernel-dtype slot))
          "the public graph carrier retains its declared width independently of node specialization"))
    (doseq [artifact [phase-one phase-two]
            :let [[slot argument] (last (map vector (:abi artifact) (:arguments artifact)))]]
      (is (= :int (:kernel-dtype slot))
          "the target-private bound retains its physical scalar representation")
      (is (= #{'n} (klaunch/expression-references argument))
          "a target-private derived bound closes over the public logical scalar"))
    (is (= 2 (count (:nodes emitted))))
    (is (every? kart/kernel-artifact? [phase-one phase-two]))
    (is (= [] (:dependencies (first source-nodes))))
    (is (= [(:id (first source-nodes))] (:dependencies (second source-nodes))))
    (is (= [:block-local :cross-block]
           (mapv #(get-in % [:effects :phase]) [phase-one phase-two])))
    (is (= [[{:value 'a :access :read} {:value partials :access :write}]
            [{:value partials :access :read} {:value result-id :access :write}]]
           (mapv #(get-in % [:effects :uses]) [phase-one phase-two])))
    (is (= (mapv :operation source-nodes) (mapv :source certificates))
        "each scheduled body retains its exact scalar SegRed phase")
    (doseq [[certificate node] (map vector certificates source-nodes)]
      (is (= certificate (segred-body/validate-against-node! certificate node graph))))
    (is (= [(first (get-in phase-one [:attributes :kernel-body :launch :group-count]))]
           (-> phase-one :attributes :kernel-body :parameters second :shape))
        "the partial output is closed over the actual checked group-count expression")
    (is (= [1] (-> phase-two :attributes :kernel-body :parameters second :shape)))
    (is (= [:checked-range :checked-range]
           (mapv #(->> % :scalar-bindings
                       (some (fn [binding]
                               (when (= '_n_bound (:parameter binding))
                                 (:conversion binding)))))
                 certificates)))
    (is (str/includes? (:source phase-two) (str (name partials) "[")))
    (is (not (re-find #"\ba\[" (:source phase-two)))
        "the cross-block target kernel must not resurrect the original reduction body")
    (is (= 2 (second (get temporary-specs partials)))
        "dynamic two-phase scratch resolves through KernelLaunch IR at call time")))

(deftest scheduled-reduction-graph-has-no-source-shaped-target-fallback
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled)
        decline (fn [& _]
                  (throw (ex-info "simulated portable coverage gap"
                                  {:reason :segred-kernel-body-declined
                                   :missing-rule :simulated
                                   :fallback :none})))]
    (doseq [target-dialect [:opencl-intel :cuda]]
      (testing (name target-dialect)
        (let [exception
              (try
                (with-redefs [segred-body/lower decline]
                  (sg/generate-kernel-graph graph :array-types {'a :double}
                                            :target-dialect target-dialect))
                nil
                (catch clojure.lang.ExceptionInfo exception exception))]
          (is (= :kernel-graph-target-lowering-missing (:reason (ex-data exception))))
          (is (= :simulated (get-in (ex-data exception)
                                    [:kernel-body-decline :missing-rule])))
          (is (= :none (:fallback (ex-data exception)))))))))

(deftest typed-stencil-emits-a-guarded-typed-artifact
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (* alpha inv-dx2
                           (+ (clojure.core/aget u (clojure.core/- i 1))
                              (* -2.0 (clojure.core/aget u i))
                              (clojure.core/aget u (clojure.core/+ i 1)))))]
                      result)
        typed (-> (typed-route/attempt
                   source :double {'du :double 'u :double}
                   {:scalar-types {'n :long 'alpha :double 'inv-dx2 :double}})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        artifact (sg/generate-segstencil-kernel-body
                  stencil
                  :array-types {'du :double 'u :double}
                  :scalar-types {'n :long 'alpha :double 'inv-dx2 :double})
        certificate (get-in artifact [:provenance :scheduled-operation])
        kernel-body (get-in artifact [:attributes :kernel-body])
        if-region (some #(when (= "IfRegion" (some-> % class .getSimpleName)) %)
                        (:operations kernel-body))
        interior-region (some #(when (= "IfRegion" (some-> % class .getSimpleName)) %)
                              (:then-operations if-region))
        loads (filterv #(= "ScalarLoad" (some-> % class .getSimpleName))
                       (:then-operations interior-region))]
    (is (kart/kernel-artifact? artifact))
    (is (scheduled-body/scheduled-kernel-body? certificate))
    (is (identical? stencil (:source certificate)))
    (is (= :scheduled-kernel-body (get-in artifact [:provenance :lowering])))
    (is (= certificate (get-in artifact [:attributes :scheduled-kernel-body])))
    (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
    (is (= '[u du alpha inv-dx2 _n_bound] (mapv :name (:abi artifact))))
    (is (= [:input :output :scalar :scalar :scalar] (mapv :kind (:abi artifact))))
    (is (= [:double :double :double :double :long] (mapv :dtype (:abi artifact))))
    (is (= [:double :double :double :double :int]
           (mapv :kernel-dtype (:abi artifact))))
    (is (= '[u du alpha inv-dx2 n] (:arguments artifact)))
    (is (= :no-write-alias (get-in artifact [:abi 0 :aliasing])))
    (is (= [{:value 'u :access :read} {:value 'du :access :write}]
           (get-in artifact [:effects :uses])))
    (is (= {:kind :stencil :boundary :dirichlet :radius 1}
           (dissoc (:effects artifact) :uses)))
    (is (= [{:parameter 'alpha :value 'alpha :dtype :double :kernel-dtype :double
             :conversion :identity}
            {:parameter 'inv-dx2 :value 'inv-dx2 :dtype :double :kernel-dtype :double
             :conversion :identity}
            {:parameter '_n_bound :value 'n :dtype :long :kernel-dtype :int
             :conversion :checked-range}]
           (:scalar-bindings certificate)))
    (is (= :portable-segstencil (get-in kernel-body [:attributes :kind])))
    (is (= 3 (count loads))
        "the interior control region must dominate every neighborhood read")))

(deftest typed-stencil-graph-preserves-its-scheduled-certificate
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (* alpha (clojure.core/aget u i)))]
                      result)
        scalar-types {'n :long 'alpha :double}
        typed (-> (typed-route/attempt source :double {'du :double 'u :double}
                                      {:scalar-types scalar-types})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        graph (kgraph/from-segops
               [stencil]
               {:inputs #{'u} :outputs #{'du} :dtype :double
                :buffer-specs {'u {:dtype :double :elements 'n}
                               'du {:dtype :double :elements 'n}}
                :scalars [(kgraph/scalar 'alpha :double)
                          (kgraph/scalar 'n :long)]})
        emitted (sg/generate-kernel-graph
                 graph :array-types {'du :double 'u :double}
                 :scalar-types scalar-types)
        artifact (get-in emitted [:nodes 0 :operation])
        certificate (get-in artifact [:provenance :scheduled-operation])]
    (is (kgraph/kernel-graph? emitted))
    (is (identical? stencil (:source certificate)))
    (is (= certificate (get-in artifact [:attributes :scheduled-kernel-body])))
    (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
    (is (= [:double :long] (mapv :dtype (:scalars emitted))))
    (is (= :long (some #(when (= '_n_bound (:name %)) (:dtype %)) (:abi artifact))))
    (is (= :int (some #(when (= '_n_bound (:name %)) (:kernel-dtype %)) (:abi artifact))))
    (is (= certificate
           (scheduled-body/validate-against-node! certificate (first (:nodes graph)) graph)))))

(deftest stencil-bound-conversion-is-proof-carrying-and-range-checked
  (let [source '(let* [result
                       (raster.par/stencil!
                        du [u] 1 :dirichlet double i n
                        (clojure.core/aget u i))]
                      result)
        scalar-types {'n :long}
        typed (-> (typed-route/attempt source :double {'du :double 'u :double}
                                      {:scalar-types scalar-types})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        scheduled (segstencil-body/schedule
                   stencil {:array-types {'du :double 'u :double}
                            :scalar-types scalar-types})
        missing-proof (update scheduled :scalar-bindings
                              (fn [bindings]
                                (mapv #(if (= '_n_bound (:parameter %))
                                         (assoc % :conversion :identity)
                                         %)
                                      bindings)))
        graph (kgraph/from-segops
               [stencil]
               {:inputs #{'u} :outputs #{'du} :dtype :double
                :buffer-specs {'u {:dtype :double :elements 'n}
                               'du {:dtype :double :elements 'n}}
                :scalars [(kgraph/scalar 'n :long)]})
        emitted (sg/generate-kernel-graph
                 graph :array-types {'du :double 'u :double}
                 :scalar-types scalar-types)]
    (try
      (scheduled-body/validate! missing-proof)
      (is false "logical long to physical int requires an explicit checked-range proof")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :scheduled-kernel-body-scalar-bindings (:reason (ex-data exception))))))
    (try
      (graph-call/make emitted {'u :resident-u 'du :resident-du}
                       {'n {:type :long :value (inc (long Integer/MAX_VALUE))}})
      (is false "a proved narrowing still rejects an out-of-range runtime extent")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-scalar-range (:reason (ex-data exception))))))))

(defn- segred-source [body-expr]
  (let [form (list 'raster.par/reduce 'acc 0.0 'j 'n body-expr)
        s (soac/par-form->soac 'result form 0 :dtype :double)
        segops (lower/lower-reduce s nil)
        operation (first segops)]
    (:source (sg/generate-segred-kernel
              operation (first (:outputs operation)) :dtype :double
              :array-types {'a :double} :scalar-types {'n :int}))))

(defn- emitted-scan-graph
  ([form] (emitted-scan-graph form {}))
  ([form opts]
   (let [result-type (or (:dtype opts) :double)
         node (soac/par-form->soac 'scan-result form 71 :dtype result-type)
         operations (lower/lower-scan node nil :dtype result-type)
         graph (lower/scan-kernel-graph node operations opts)]
     (sg/generate-scan-kernel-graph graph
                                    :scalar-types (:scalar-types opts)))))

(defn- kernel-body-operations
  [artifact]
  (letfn [(walk [operations]
            (mapcat
             (fn [operation]
               (cons operation
                     (concat (walk (or (:operations operation) []))
                             (walk (or (:then-operations operation) []))
                             (walk (or (:else-operations operation) [])))))
             operations))]
    (walk (get-in artifact [:attributes :kernel-body :operations]))))

(defn- operation-kind [operation]
  (some-> operation class .getSimpleName))

(deftest segscan-graph-emits-one-verified-artifact-per-scheduled-node
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double
                                   (+ acc (* scale (clojure.core/aget values i))))
                 {:array-types {'values :double 'out :double}
                  :scalar-types {'scale :double}})
        [intra block carry] (mapv :operation (:nodes emitted))]
    (is (kgraph/kernel-graph? emitted))
    (is (every? kart/kernel-artifact? [intra block carry]))
    (is (= [:intra-block :block-scan :carry-in]
           (mapv #(get-in % [:attributes :phase]) [intra block carry])))
    (is (every? #(= :portable-segscan
                    (get-in % [:attributes :kernel-body :attributes :kind]))
                [intra block carry]))
    (testing "stage 1 owns both the result and graph temporary in one exact ABI"
      (is (= [:temporary :result :operand :parameter :bound]
             (mapv :role (:abi intra))))
      (is (= '[scale n] (take-last 2 (:arguments intra))))
      (let [operations (kernel-body-operations intra)]
        (is (some #(and (= "ScalarLoad" (operation-kind %))
                        (= 'values (:buffer %)))
                  operations))
        (is (some #(and (= "ScalarCompute" (operation-kind %))
                        (= :* (get-in % [:expression :op])))
                  operations))))
    (testing "block totals are scanned in one chunked workgroup for unbounded symbolic sizes"
      (is (= [1] (get-in block [:launch :group-count])))
      (let [operations (kernel-body-operations block)
            loop (some #(when (= "ForLoop" (operation-kind %)) %) operations)
            safe-bound (some #(when (and (= "ScalarCompute" (operation-kind %))
                                         (= 'scan-safe-bound (get-in % [:result :id])))
                                %)
                             operations)
            safe-expression (:expression safe-bound)]
        (is (= [0 'scan-safe-bound 256 :ordered]
               [(:lower loop) (:upper loop) (:step loop)
                (get-in loop [:attributes :association])]))
        (is (= :min (:op safe-expression)))
        (is (= :max (get-in safe-expression [:arguments 0 :op])))
        (is (= '_n_bound (get-in safe-expression [:arguments 0 :arguments 0]))
            "the in-body clamp, rather than an ABI role assertion, proves the loop extent")))
    (testing "carry consumes the scanned total of the preceding block"
      (let [loads (filter #(= "ScalarLoad" (operation-kind %))
                          (kernel-body-operations carry))]
        (is (= #{(first (map :id (:temporaries emitted))) 'out}
               (set (map :buffer loads))))))
    (testing "entry-point names are valid C identifiers"
      (is (every? #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" (:kernel-name %))
                  [intra block carry])))
    (let [[intra-node block-node carry-node] (:nodes emitted)]
      (is (= [(:id intra-node)] (:dependencies block-node)))
      (is (= (mapv :id [intra-node block-node]) (:dependencies carry-node))
          "target lowering preserves the verified schedule"))))

(deftest typed-scan-schedule-target-lowers-without-a-legacy-soac-node
  (let [source '(let* [result (raster.par/scan out acc 0.0 i n float
                                               (+ acc (clojure.core/aget values i)))]
                      result)
        typed (:program (typed-route/attempt source :float
                                             {'values :float 'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'values :float 'out :float}}))
        graph (get-in scheduled [:equations 0 :attributes :kernel-graph])
        emitted (sg/generate-kernel-graph graph)]
    (is (= :typed-soac (get-in scheduled [:provenance :source-dialect])))
    (is (= 3 (count (:nodes emitted))))
    (is (every? kart/kernel-artifact? (map :operation (:nodes emitted))))
    (is (= [:intra-block :block-scan :carry-in]
           (mapv #(get-in % [:operation :attributes :phase]) (:nodes emitted))))))

(deftest segscan-artifact-abi-preserves-mixed-input-storage
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double
                                   (+ acc (double (clojure.core/aget values i))))
                 {:array-types {'values :int 'out :double}})
        intra (get-in emitted [:nodes 0 :operation])
        values-slot (first (filter #(= 'values (:name %)) (:abi intra)))]
    (is (= :int (:dtype values-slot)))
    (is (= :int (:kernel-dtype values-slot)))
    (is (re-find #"__global const int\* restrict values" (:source intra)))))

(deftest small-segscan-emits-the-degenerate-one-artifact-graph
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i 64 double (+ acc (aget values i))))
        artifact (get-in emitted [:nodes 0 :operation])]
    (is (= 1 (count (:nodes emitted))))
    (is (= :single (get-in artifact [:attributes :phase])))
    (is (= [1] (get-in artifact [:launch :group-count])))
    (is (not (re-find #"block_totals" (:source artifact))))))

(deftest segscan-carry-emits-the-certified-combine
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 1.0 i n double (* acc (aget values i))))
        carry (get-in emitted [:nodes 2 :operation])
        combines (for [operation (kernel-body-operations carry)
                       :when (= "ScalarCompute" (operation-kind operation))]
                   (get-in operation [:expression :op]))]
    (is (= [:*] (vec combines))
        "carry propagation must use the certified monoid, not a hard-coded addition")))

(deftest integral-segscan-carries-certified-wrapping-through-every-combine
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc (int 0) i n int
                                   (unchecked-add-int acc (aget values i)))
                 {:dtype :int
                  :array-types {'values :int 'out :int}
                  :scalar-types {'n :int}})
        combines (for [node (:nodes emitted)
                       operation (kernel-body-operations (:operation node))
                       :when (and (= "ScalarCompute" (operation-kind operation))
                                  (= :+ (get-in operation [:expression :op]))
                                  (= :int (get-in operation [:expression :result-type])))]
                   (:expression operation))]
    (is (seq combines))
    (is (every? #(= :wrap (get-in % [:options :overflow])) combines)
        "every reassociated integer combine must retain the certified modulo-2^N policy")))

(deftest segscan-carry-separates-its-launch-width-from-the-scanned-block-width
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              73)
        operations (lower/lower-scan node nil :dtype :float)
        scan-width (get-in operations [0 :grid :block-size])
        carry-width (quot scan-width 2)
        operations (assoc-in operations [2 :grid]
                             (segop/->KernelGrid
                              (klaunch/ceil-div 'n carry-width)
                              carry-width 0))
        graph (lower/scan-kernel-graph
               node operations {:array-types {'values :float 'out :float}})
        carry (get-in (sg/generate-scan-kernel-graph graph) [:nodes 2 :operation])
        scan-block (some #(when (= 'scan-block (:id %)) %)
                         (get-in carry [:attributes :kernel-body :indices]))]
    (is (= [carry-width] (get-in carry [:launch :workgroup-size])))
    (is (= scan-width
           (get-in carry [:attributes :kernel-body :schedule :scan-block-size])))
    (is (= (kernel-body/expression :floor-div 'scan-index scan-width)
           (:expression scan-block)))))

;; ================================================================
;; Silently-ignored-information family: a SegRed combine that the
;; extractor only partially models must FAIL LOUD, not miscompile.
;; ================================================================

(deftest segred-nary-combine-rejected
  (testing "a variadic (+ acc x y) combine is rejected, not silently summing only x"
    ;; Before the fix, op-args extraction took a0=acc a1=x and DROPPED y — emitting a
    ;; kernel that reduced with just x. A segmented reduce combine is binary (op acc elem).
    (is (thrown-with-msg?
         Exception #"certified associative reduction"
         (segred-source '(+ acc (clojure.core/aget a j) (clojure.core/aget b j))))))
  (testing "the ordinary binary combine still lowers unchanged"
    (let [source (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (re-find #"tree_combined_.* = .* \+ " source))
      (is (re-find #"output\[" source)))))

(deftest segred-multistatement-lambda-rejected
  (testing "a reduce lambda with a multi-statement let body is rejected, not (last bdy)-dropped"
    ;; The single-aset-void store-drop shape on the reduce side: (last bdy) silently dropped
    ;; the earlier body forms.
    (is (thrown-with-msg?
         Exception #"certified associative reduction"
         (segred-source '(let* [t (clojure.core/aget a j)] (+ acc t) (+ acc t)))))))

(deftest segred-devirtualized-aget-lowers-to-subscript
  (testing "the parametric (.invk aget-impl …) shape — the qlinear-k side of #55"
    (let [aget-invk (with-meta (list '.invk 'raster.arrays/aget_m_doubles_long-impl 'a 'j)
                      {:raster.op/original 'raster.arrays/aget
                       :raster.type/tag 'double})
          plus-invk (with-meta (list '.invk 'raster.numeric/_plus__m_double_double-impl
                                     'acc aget-invk)
                      {:raster.op/original 'raster.numeric/+
                       :raster.type/tag 'double})
          src (segred-source plus-invk)]
      (is (not (re-find #"gpufn_aget" src))
          "devirtualized aget must normalize to a subscript, not a helper call")
      (is (re-find #"a\[" src)))))

(deftest segred-bare-aget-lowers-to-subscript
  (testing "the typed clojure.core/aget shape — the decoder-gpu side of #55"
    (let [src (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (not (re-find #"gpufn_aget" src)))
      (is (re-find #"a\[" src)))))

(deftest segred-emits-complete-ordered-typed-abi
  (let [product (with-meta '(* scale (clojure.core/aget a i))
                  {:raster.type/tag 'float})
        form (with-meta (list 'raster.par/reduce 'acc 0.0 'i 'n
                              (list '+ (list 'float 'acc) product))
               {:raster.type/elem-type :float})
        s (soac/par-form->soac 'result form 0 :dtype :float)
        segred (first (lower/lower-reduce s nil))
        physical-output (first (:outputs segred))
        k (sg/generate-segred-kernel segred (first (:outputs segred)) :dtype :float
                                     :scalar-types {'scale :float 'n :int})]
    (is (kart/kernel-artifact? k))
    (testing "signature, ABI and compiler values have one identical order"
      (is (= ['a physical-output 'scale '_n_bound] (mapv :name (:abi k))))
      (is (= [:input :output :scalar :scalar] (mapv :kind (:abi k))))
      (is (= [:float :float :float :int] (mapv :kernel-dtype (:abi k))))
      (is (= [:operand :result :parameter :bound] (mapv :role (:abi k))))
      (is (= ['a physical-output 'scale 'n] (:arguments k)))
      (is (= (kabi/signature-shape (:abi k))
             (kabi/source-signature-shape (:kernel-name k) (:source k))))
      (is (= :no-write-alias (get-in k [:abi 0 :aliasing]))))
    (testing "launch and semantic origin are part of the value"
      (is (klaunch/launch-spec? (:launch k)))
      (is (= [256] (get-in k [:launch :workgroup-size])))
      (is (= 1024 (get-in k [:launch :shared-memory-bytes])))
      (is (= :kernel-body (get-in k [:provenance :dialect])))
      (is (= (:id segred) (get-in k [:provenance :segop-id])))
      (is (= :scheduled-kernel-body (get-in k [:provenance :lowering])))
      (is (= :float (get-in k [:attributes :dtype])))
      (is (= :kernel-body (get-in k [:attributes :emission-route])))
      (is (some? (get-in k [:attributes :kernel-body])))
      (is (nil? (get-in k [:attributes :n-phases]))
          "one phase artifact does not claim that it owns two phases")))
  (testing "the artifact retains a physical result identity; only the host marker substitutes nil"
    (let [form '(raster.par/reduce acc 0.0 i n (+ acc (clojure.core/aget a i)))
          s (soac/par-form->soac 'result form 0 :dtype :float)
          operation (first (lower/lower-reduce s nil))
          physical-output (first (:outputs operation))
          k (sg/generate-segred-kernel operation nil :dtype :float
                                       :scalar-types {'n :int})]
      (is (= ['a physical-output '_n_bound] (mapv :name (:abi k))))
      (is (= ['a physical-output 'n] (:arguments k))))))

(deftest one-scalar-segred-schedule-emits-through-every-c-family-fixture
  (let [operation (first (lower/lower-reduce
                          (soac/par-form->soac
                           'result
                           '(raster.par/reduce acc 0.0 i n
                                               (+ acc (clojure.core/aget a i)))
                           90 :dtype :float)
                          nil :dtype :float))]
    (doseq [[target expected]
            [[:opencl-portable :opencl-c]
             [:cuda :cuda-c]
             [:hip :hip-cpp]]]
      (testing (name target)
        (let [artifact (sg/generate-segred-kernel
                        operation nil :dtype :float :target-dialect target
                        :scalar-types {'n :int})
              certificate (get-in artifact [:provenance :scheduled-operation])]
          (is (= expected (:target artifact)))
          (is (identical? operation (:source certificate)))
          (is (= artifact
                 (scheduled-body/validate-artifact-projection!
                  certificate artifact))))))))

(deftest scalar-segred-graph-retains-checked-product-extent-alias
  (let [source (list 'let*
                     ['count (with-meta '(clojure.core/* rows width)
                               {:raster.type/tag 'long})
                      'result '(raster.par/reduce acc 0.0 i count
                                                 (+ acc (clojure.core/aget a i)))]
                     'result)
        options {:dtype :double :resident-reductions? true :array-types {'a :double}
                 :scalar-types {'rows :long 'width :long}}
        route-result (typed-route/attempt source :double {'a :double} options)
        _ (when-not (:program route-result)
            (throw (ex-info "product-alias fixture must enter the typed route" route-result)))
        typed (:program route-result)
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        numerical (filterv (comp seq :operations) (:equations scheduled))
        {:keys [algorithm body graph]} (equation-graph/make-for-equations scheduled numerical)
        definitions (equation-graph/derived-scalar-expressions
                     (:values body)
                     (take-while #(true? (get-in % [:attributes :host-only])) (:equations body)))
        emit (fn [owner-body]
               (sg/generate-kernel-graph
                graph :array-types {'a :double}
                :scalar-types (into {} (keep (fn [[id value]]
                                              (when (empty? (:shape value)) [id (:dtype value)])))
                                    (:values body))
                :scheduled-equation-algorithm algorithm :scheduled-equation-body owner-body))]
    (is (contains? definitions 'count) "a validated leading host scalar supplies the alias")
    (is (every? kart/kernel-artifact? (map :operation (:nodes (emit body))))
        "source/body/graph spellings must close over their checked extent definition")
    (doseq [[expected-reason invalid-body]
            [[:parallel-program-use-before-definition (update body :equations #(vec (rest %)))]
             [:scheduled-equation-prefix (assoc-in body [:values 'count :dtype] :int)]
             [:parallel-program-equation-values
              (assoc-in body [:equations 0 :operands] ['width 'width])]]]
      (try
        (emit invalid-body)
        (is false "invalid owner definitions cannot qualify the graph")
        (catch clojure.lang.ExceptionInfo error
          (is (= expected-reason (:reason (ex-data error)))))))))

(deftest scalar-segred-schedule-rejects-a-forged-graph-source
  (let [source '(let* [result (raster.par/reduce acc 0.0 i n
                                                 (+ acc (clojure.core/aget a i)))]
                      result)
        options {:dtype :double :array-types {'a :double} :scalar-types {'n :long}}
        typed (:program (typed-route/attempt source :double {'a :double} options))
        algorithm (get-in typed [:equations 0 :algorithm])
        scheduled-program (:form (segop-lower/segop-lower-pass typed options))
        graph (equation-graph/make algorithm scheduled-program)
        node (last (:nodes graph))
        operation (:operation node)
        certificate (segred-body/schedule operation (first (:outputs operation)) options)
        forged-node (assoc node :operation (assoc operation :id :forged-reduction))
        forged-storage (update graph :outputs
                               #(mapv (fn [buffer] (assoc buffer :elements 2)) %))
        forged-certificates
        [(assoc-in certificate [:body :attributes :identity] 1.0)
         (assoc-in certificate [:body :schedule :reduction-operator] :*)
         (assoc-in certificate [:numerics :policy] :forged-tree)]]
    (is (identical? operation (:source certificate)))
    (try
      (sg/generate-segred-kernel operation 'forged-result
                                 :dtype :double :array-types {'a :double})
      (is false "production scalar emission cannot replace the semantic output identity")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :physical-output-remap
               (get-in (ex-data exception)
                       [:kernel-body-decline :missing-rule])))))
    (try
      (segred-body/validate-against-node! certificate forged-node graph)
      (is false "a boundary-compatible but different SegRed must not reuse the certificate")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :scheduled-kernel-body-source (:reason (ex-data exception))))))
    (try
      (segred-body/validate-against-node! certificate node forged-storage)
      (is false "the graph result allocation must equal the scheduled output group count")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :storage-extent (:missing-rule (ex-data exception))))))
    (try
      (sg/generate-kernel-graph forged-storage
                                :array-types {'a :double}
                                :scalar-types {'n :long})
      (is false "production graph emission must apply the SegRed/node storage validator")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :storage-extent
               (get-in (ex-data exception) [:kernel-body-decline :missing-rule])))))
    (doseq [forged forged-certificates]
      (try
        (segred-body/validate-against-node! forged node graph)
        (is false "a body or numerical certificate cannot be changed independently of source")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :schedule-source (:missing-rule (ex-data exception)))))))))

(deftest scalar-segred-schedule-rejects-forged-phase-and-tree-facts
  (let [operation (first (lower/lower-reduce
                          (soac/par-form->soac
                           'result
                           '(raster.par/reduce acc 0.0 i 4096
                                               (+ acc (clojure.core/aget a i)))
                           92 :dtype :float)
                          nil :dtype :float))
        options {:array-types {'a :float}}
        missing-rule
        (fn [candidate]
          (try
            (segred-body/schedule candidate nil options)
            nil
            (catch clojure.lang.ExceptionInfo exception
              (:missing-rule (ex-data exception)))))
        cases
        [[:fused-map-representation (assoc operation :lambda '(identity element))]
         [:phase-level (assoc operation :level (segop/->SegLevel :block :none))]
         [:grid-shared-memory (assoc-in operation [:grid :shared-mem-bytes] 0)]
         [:schedule-grid (assoc-in operation [:schedule :workgroup-size] 128)]
         [:schedule-grid (assoc-in operation [:schedule :attributes :group-count] 1)]
         [:schedule-grid (assoc-in operation [:schedule :numerical-mode :overflow] :wrap)]
         [:physical-phase (assoc-in operation [:reduction :attributes :physical-phase]
                                    :cross-block)]]]
    (doseq [[expected candidate] cases]
      (testing (name expected)
        (is (= expected (missing-rule candidate)))))
    (let [grid (assoc (:grid operation) :num-blocks (klaunch/sum 1 1))
          forged (assoc operation :grid grid
                          :schedule (segred-body/scalar-workgroup-tree-schedule
                                     (:reduction operation) grid (:phase operation)))]
      (is (= :launch-grid (missing-rule forged))
          "an internally self-consistent but unrelated launch expression is not a proof"))))

(deftest scalar-segred-max-int-bound-uses-wide-overflow-free-schedule-arithmetic
  (let [operation (first (lower/lower-reduce
                          (soac/par-form->soac
                           'result
                           (list 'raster.par/reduce 'acc 0.0 'i Integer/MAX_VALUE
                                 '(+ acc (clojure.core/aget a i)))
                           93 :dtype :float)
                          nil :dtype :float))
        kernel-body (:kernel-body
                     (segred-body/lower operation nil
                                        :dtype :float :array-types {'a :float}))
        casts (filterv #(instance? raster.compiler.ir.kernel_body.IndexCast
                                   (:expression %))
                       (:indices kernel-body))
        loop (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                            (:operations kernel-body)))
        index-expressions (into {} (map (juxt :id :expression) (:indices kernel-body)))]
    (is (= '[wide-bound wide-group-index wide-group-count wide-local-index]
           (mapv :id casts)))
    (is (every? #(= :long (get-in % [:expression :dtype])) casts))
    (is (= :long (get-in loop [:index :type])))
    (is (= :ceil-div (get-in index-expressions ['group-chunk :op]))
        "the checked index algebra handles zero without forming n-1 or n+groups-1")
    (is (= :add (:op (get index-expressions 'group-end))))))

(deftest completed-scalar-reduction-transform-is-terminal-and-numerically-explicit
  (let [form (with-meta
               '(raster.par/reduce acc 0.0 i 32 (+ acc (clojure.core/aget a i)))
               {:raster.type/elem-type :float})
        base (first (lower/lower-reduce (soac/par-form->soac 'result form 91 :dtype :float) nil
                                        :dtype :float))
        transform (kernel-body/->ScalarRegion
                   '[completed scale] '(clojure.core/* completed scale) [] :float)
        terminal (-> base
                     (assoc-in [:reduction :attributes :result-region] transform)
                     (assoc :scalars #{'scale}))
        certificate (segred-body/schedule
                     terminal nil {:array-types {'a :float}
                                   :scalar-types {'scale :float}})]
    (is (= :single (:phase terminal)))
    (is (= {:kind :typed-scalar-region
            :policy :same-typed-ssa-evaluation-order
            :input-dtype :float :result-dtype :float}
           (get-in certificate [:numerics :result-transform])))
    (is (= :certified-workgroup-tree (get-in certificate [:numerics :policy])))
    (let [nonterminal (assoc terminal
                             :phase :block-local
                             :level (segop/->SegLevel :block :virtual)
                             :schedule (segred-body/scalar-workgroup-tree-schedule
                                        (:reduction terminal) (:grid terminal) :block-local))]
      (try
        (segred-body/schedule nonterminal nil
                             {:array-types {'a :float}
                              :scalar-types {'scale :float}})
        (is false "a block-local phase must not apply the completed transform")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :nonterminal-result-transform
                 (:missing-rule (ex-data exception)))))))))

(deftest completed-conversion-separates-terminal-storage-from-reduction-state
  (doseq [[source target] [[:float :double] [:double :float]]]
    (let [form (with-meta
                 '(raster.par/reduce acc 0.0 i 32 (+ acc (clojure.core/aget a i)))
                 {:raster.type/elem-type source})
          base (first (lower/lower-reduce (soac/par-form->soac 'result form 91 :dtype source)
                                          nil :dtype source))
          transform (kernel-body/->ScalarRegion
                     '[completed] (list (symbol (name target)) 'completed) [] target)
          terminal (assoc-in base [:reduction :attributes :result-region] transform)
          options {:array-types {'a source 'result target}}
          certificate (segred-body/schedule terminal nil options)
          parameters (get-in certificate [:body :parameters])]
      (is (= target (:dtype (first (filter #(= :output (:kind %)) parameters)))))
      (is (= source (:dtype (first (filter #(= :input (:kind %)) parameters)))))
      (is (= source (get-in certificate [:numerics :accumulator-dtype])))
      (is (= target (get-in certificate [:numerics :result-transform :result-dtype])))
      (is (= source (:dtype (first (get-in certificate [:body :allocations])))))
      (when (= :float source)
        ;; The historical 0.25 Double literal makes this addition genuinely Double.
        ;; Preserve it as a positive for the independently certified terminal class.
        (let [double-arithmetic (assoc-in terminal
                                         [:reduction :attributes :result-region :expression]
                                         '(double (+ completed 0.25)))
              arithmetic-certificate (segred-body/schedule double-arithmetic nil options)
              operations (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                 (tree-seq coll? seq (:body arithmetic-certificate)))]
          (is (= :float (get-in arithmetic-certificate [:numerics :accumulator-dtype])))
          (is (= :double (get-in arithmetic-certificate
                                 [:numerics :result-transform :result-dtype])))
          (is (some #(and (= :+ (get-in % [:expression :op]))
                          (= :double (get-in % [:result :type]))) operations))))
      (doseq [candidate [base
                         (assoc-in terminal [:reduction :attributes :result-region :expression]
                                   (list (symbol (name target))
                                         (list '+ 'completed
                                               (if (= :float source) (float 0.25) 0.25))))]]
        (try
          (segred-body/schedule candidate nil options)
          (is false "mixed output storage must preserve the source intermediate precision")
          (catch clojure.lang.ExceptionInfo error
            (is (= :uniform-scalar-storage (:missing-rule (ex-data error))))))))))

(deftest completed-arithmetic-cannot-retag-checked-operands
  (let [form (with-meta
               '(raster.par/reduce acc 0.0 i 32 (+ acc (clojure.core/aget a i)))
               {:raster.type/elem-type :float})
        base (first (lower/lower-reduce (soac/par-form->soac 'result form 91 :dtype :float)
                                       nil :dtype :float))
        options {:array-types {'a :float 'result :double}
                 :scalar-types {'scale :float 'counter :long}}
        schedule (fn [parameters expression]
                   (segred-body/schedule
                    (assoc-in (assoc base :scalars (set (rest parameters)))
                              [:reduction :attributes :result-region]
                              (kernel-body/->ScalarRegion parameters expression [] :double))
                    nil options))]
    (testing "genuine Double arithmetic remains admitted with checked Float operands"
      (is (= :double (get-in (schedule '[completed counter]
                                      '(double (/ completed (double counter))))
                            [:numerics :result-transform :result-dtype]))))
    (doseq [metadata-key [:raster.type/tag :tag]
            [parameters operand other]
            [['[completed] 'completed (float 1)]
             ['[completed scale] 'scale 'completed]
             ['[completed counter] 'counter 'completed]]]
      (let [tagged (with-meta operand {metadata-key 'double})
            expression (list 'double (list 'clojure.core/+ tagged other))]
        (testing (str "checked operand " operand " overrides " metadata-key)
          (try
            (schedule parameters expression)
            (is false "metadata cannot turn checked Float arithmetic into Double arithmetic")
            (catch clojure.lang.ExceptionInfo error
              (is (= :uniform-scalar-storage (:missing-rule (ex-data error)))))))))))

(deftest unproved-affine-loads-decline-the-pointwise-reduction-route
  (let [form '(raster.par/reduce acc 0.0 i n
                                 (+ acc (clojure.core/aget a (+ i offset))))
        node (soac/par-form->soac 'result form 22 :dtype :float)
        operation (first (lower/lower-reduce node nil :dtype :float))]
    (try
      (sg/generate-segred-kernel
       operation nil :dtype :float :scalar-types {'offset :int 'n :int})
      (is false "an n-element buffer does not prove i+offset is in bounds")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-graph-target-lowering-missing
               (:reason (ex-data exception))))
        (is (= :indexed-load
               (get-in (ex-data exception)
                       [:kernel-body-decline :missing-rule])))))))

(deftest mixed-floating-reduction-arithmetic-is-explicit-typed-ssa
  (let [product (with-meta
                  '(* (double scale) (clojure.core/aget a i))
                  {:raster.type/tag 'double})
        form (with-meta
               (list 'raster.par/reduce 'acc 0.0 'i 'n
                     (list '+ 'acc (list 'float product)))
               {:raster.type/elem-type :float})
        node (soac/par-form->soac 'result form 88 :dtype :float)
        operation (first (lower/lower-reduce node :ze:0 :dtype :float))
        artifact (sg/generate-segred-kernel
                  operation nil :dtype :float
                  :scalar-types {'scale :double 'n :long})
        loop-body (:operations (first (:operations
                                       (get-in artifact [:attributes :kernel-body]))))
        casts (keep (fn [operation]
                      (let [expression (:expression operation)]
                        (when (= :cast (:op expression))
                          [(:result-type expression) (:options expression)])))
                    loop-body)]
    (is (= [[:double {:rounding :exact :overflow :exact}]
            [:float {:rounding :nearest-even :overflow :ieee}]]
           casts))
    (is (= :double (some #(when (= 'scale (:name %)) (:kernel-dtype %))
                         (:abi artifact))))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))))

(deftest reduction-scalar-typing-declines-uncertified-language-semantics
  (let [options {:index 'i :coordinate 'element-index :dtype :float
                 :arrays #{'a} :scalars #{'scale 'counter}
                 :scalar-types {'scale :double 'counter :long}}
        decline (fn [expression]
                  (try
                    (segred-body/lower-element-operations expression options)
                    nil
                    (catch clojure.lang.ExceptionInfo exception
                      (ex-data exception))))]
    (testing "mixed arithmetic needs its retained walker/TypedClojure result fact"
      (is (= :scalar-result-dtype
             (:missing-rule
              (decline '(* scale (clojure.core/aget a i)))))))
    (testing "even uniform Float operands do not identify clojure.core versus typed arithmetic"
      (is (= :scalar-result-dtype
             (:missing-rule
              (decline '(* (clojure.core/aget a i)
                           (clojure.core/aget a i)))))))
    (testing "checked integral casts are not reinterpreted as wrap or saturation"
      (is (= :checked-scalar-cast (:missing-rule (decline '(int scale)))))
      (is (= :checked-scalar-cast (:missing-rule (decline '(byte 128))))))
    (testing "integral value arithmetic waits for an explicit overflow contract"
      (is (= :integral-scalar-arithmetic
             (:missing-rule
              (decline (with-meta '(+ counter 1) {:raster.type/tag 'long}))))))
    (testing "literal and expression results cannot be silently narrowed to the accumulator"
      (is (= {:missing-rule :element-result-dtype :element-dtype :double}
             (select-keys (decline 0.1) [:missing-rule :element-dtype])))
      (is (= :element-result-dtype
             (:missing-rule
              (decline (with-meta '(* scale (clojure.core/aget a i))
                         {:raster.type/tag 'double}))))))
    (testing "beta reduction cannot erase a typed local conversion"
      (let [local (with-meta 'local {:raster.type/tag 'float})]
        (is (= :typed-scalar-binding-conversion
               (:missing-rule
                (decline (list 'let* [local 0.1] (list 'double local))))))))))

;; ================================================================
;; Horizontally-fused multi-output SegMap: the SECONDARY output (written
;; only via a side-effect aset in the fused lambda) must be a NON-const
;; __global array param — never a scalar. Before the fix it was declared
;; `float hfuse_out__N` (a scalar) while the body indexed it as an array:
;; a broken kernel, and the extraction layer then eval'd the bare buffer
;; symbol on the host (`Unable to resolve symbol: hfuse_out__N`).
;; ================================================================

(deftest segmap-fused-secondary-output-is-array-param
  (let [form '(raster.par/map! hout1 i n float
                               (do (raster.arrays/aset hout2 i (float (* (clojure.core/aget d i)
                                                                         (clojure.core/aget a i))))
                                   (* (clojure.core/aget d i) (clojure.core/aget b i))))
        s (soac/par-form->soac 'da form 0)
        segmap (first (lower/lower-map s nil :dtype :float))
        k (sg/generate-segmap-kernel segmap 'hout1 :dtype :float)]
    (testing "secondary output is an array param, not a scalar param"
      (is (kart/kernel-artifact? k))
      (is (some #{'hout2} (kart/attribute k :array-params)))
      (is (not (some #{'hout2} (kart/attribute k :scalar-params))))
      (is (some #{'hout2} (kart/attribute k :written-arrays))))
    (testing "declared __global and NON-const in the C signature"
      (is (re-find #"__global float\* restrict hout2" (:source k)))
      (is (not (re-find #"const float\* restrict hout2" (:source k)))))
    (testing "written via subscript in the body"
      (is (re-find #"hout2\[" (:source k))))
    (testing "the ABI preserves secondary-output, primary-output, scalar and bound order"
      (is (= '[a b d hout2 hout1 _n_bound] (mapv :name (:abi k))))
      (is (= [:input :input :input :output :output :scalar]
             (mapv :kind (:abi k))))
      (is (= '[a b d hout2 hout1 n] (:arguments k))))))

(deftest segmap-abi-preserves-integer-scalar-type
  (let [form '(raster.par/map! out i n float
                               (clojure.core/aget a (clojure.core/+ i offset)))
        s (soac/par-form->soac 'out form 1)
        segmap (first (lower/lower-map s nil :dtype :float))
        k (sg/generate-segmap-kernel segmap 'out :dtype :float
                                     :scalar-types {'offset :int})]
    (is (= '[a out offset _n_bound] (mapv :name (:abi k))))
    (is (= [:float :float :int :int] (mapv :dtype (:abi k))))
    (is (re-find #"int offset" (:source k)))
    (is (= '[a out offset n] (:arguments k)))))

(deftest scheduled-segmap-fails-closed-at-the-kernel-body-boundary
  (let [form '(raster.par/map! out i n float
                               (clojure.core/aget a i))
        operation (-> (soac/par-form->soac 'out form 2)
                      (lower/lower-map nil :dtype :float)
                      first)
        failure
        (with-redefs [segmap-body/lower
                      (fn [& _]
                        (throw (ex-info "simulated portable coverage gap"
                                        {:reason :segmap-kernel-body-declined
                                         :missing-rule :simulated
                                         :fallback :none})))]
          (try
            (sg/generate-scheduled-segmap-kernel operation :dtype :float)
            nil
            (catch clojure.lang.ExceptionInfo exception
              (ex-data exception))))]
    (is (= :segmap-kernel-body-declined (:reason failure)))
    (is (= :simulated (:missing-rule failure)))
    (is (= :none (:fallback failure)))))

(defn- quotient-neighbour-operation []
  ;; Complete retained restriction arithmetic: quotient, subtraction-based
  ;; remainder, doubled coordinates, width product and all four neighbour loads.
  ;; Width is its actual Long ABI capture; this fixture infers no relation to d.
  (segop/->SegMap
   905 (segop/make-seg-space 'flat 'n) (segop/->SegLevel :thread :virtual)
   '(let [q (quot (long flat) divisor)
          r (- (long flat) (* q divisor))
          row2 (* (long 2) q)
          col2 (* (long 2) r)
          k (+ (* row2 width) col2)]
      (* 0.25 (+ (clojure.core/aget fine k)
                 (clojure.core/aget fine (+ k 1))
                 (clojure.core/aget fine (+ k width))
                 (clojure.core/aget fine (+ k width 1)))))
   {:locals [{:id 'q :dtype :long :init '(quot (long flat) divisor)}
             {:id 'r :dtype :long :init '(- (long flat) (* q divisor))}
             {:id 'row2 :dtype :long :init '(* (long 2) q)}
             {:id 'col2 :dtype :long :init '(* (long 2) r)}
             {:id 'k :dtype :long :init '(+ (* row2 width) col2)}]
    :result '(* 0.25 (+ (clojure.core/aget fine k)
                       (clojure.core/aget fine (+ k 1))
                       (clojure.core/aget fine (+ k width))
                       (clojure.core/aget fine (+ k width 1))))}
   #{'fine} #{'output} #{'divisor 'width}
   (segop/->KernelGrid 1 256 0) :double 'output nil))

(deftest guarded-quotient-and-complete-neighbour-chain-emit-portably
  (let [operation (quotient-neighbour-operation)
        options {:scalar-types {'n :long 'divisor :long 'width :long}
                 :array-types {'fine :double 'output :double}}
        scheduled (segmap-body/schedule operation options)
        checked (:body scheduled)
        result (try
                 {:artifact (sg/generate-scheduled-segmap-kernel
                             operation :scalar-types (:scalar-types options)
                             :array-types (:array-types options)
                             :target-dialect :opencl-portable)}
                 (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
    (is (= checked (kernel-body/validate! checked))
        "prerequisite: complete quotient and neighbour chain is checked KernelBody IR")
    (is (nil? (:error result))
        (str "the full guarded source must emit with retained conditional obligations: " (:error result)))
    (when-let [artifact (:artifact result)]
      (is (seq (:preconditions artifact)))
      (is (= artifact (kart/validate! artifact))))))

(deftest quotient-neighbour-load-offsets-have-zero-safe-host-obligations
  (let [artifact (sg/generate-scheduled-segmap-kernel
                  (quotient-neighbour-operation)
                  :scalar-types {'n :long 'divisor :long 'width :long}
                  :array-types {'fine :double 'output :double}
                  :target-dialect :opencl-portable)
        arguments (fn [n divisor width]
                           (mapv (fn [slot]
                                   (case (:name slot)
                                     fine ::fine output ::output
                                     divisor {:type :long :value divisor}
                                     width {:type :long :value width}
                                     _n_bound {:type :long :value n})) (:abi artifact)))]
    (doseq [divisor [Long/MIN_VALUE 0 Long/MAX_VALUE]]
      (is (kcall/validate-preconditions! artifact (arguments 0 divisor Long/MAX_VALUE))
          "empty work makes every unused neighbour term zero before checked evaluation"))
    (is (thrown? ArithmeticException
                 (kcall/validate-preconditions! artifact (arguments 1 Long/MAX_VALUE Long/MAX_VALUE)))
        "actual emitted load k+width+1 must reject active overflow even when q=r=k=0")))

(defn- quotient-neighbour-artifact [target]
  (sg/generate-scheduled-segmap-kernel
   (quotient-neighbour-operation)
   :scalar-types {'n :long 'divisor :long 'width :long}
   :array-types {'fine :double 'output :double} :target-dialect target))

(deftest quotient-address-obligations-are-independent-and-transported
  (doseq [target [:opencl-portable :opencl-intel :cuda :hip]]
    (let [artifact (quotient-neighbour-artifact target)
          checked (get-in artifact [:attributes :kernel-body])
          scheduled (get-in artifact [:attributes :scheduled-kernel-body])
          conditions (:preconditions artifact)]
      (is (= conditions (:required-preconditions
                         (scalar-emitter/emit-scalar-module
                          "quotient_module" checked {:target-dialect target}))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (scalar-emitter/emit-scalar-kernel
                    "quotient_raw" checked {:target-dialect target})))
      (doseq [index (range (count conditions))]
        (let [missing (into (subvec conditions 0 index) (subvec conditions (inc index)))]
          (is (thrown? clojure.lang.ExceptionInfo
                       (kernel-body/validate!
                        (assoc-in checked [:attributes :required-preconditions] missing)))
              (str "independent body reconstruction requires intermediate/address condition " index))
          (is (thrown? clojure.lang.ExceptionInfo
                       (scheduled-body/validate! (assoc scheduled :preconditions missing))))
          (is (thrown? clojure.lang.ExceptionInfo
                       (kart/validate! (assoc artifact :preconditions missing))))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kernel-body/validate!
                    (assoc-in checked [:masks 0 :predicates 0 :op] :lte)))
          "quotient facts require the actual dominating upper-domain mask")
      (is (thrown? clojure.lang.ExceptionInfo
                   (kernel-body/validate!
                    (assoc checked :operations (:operations (first (:operations checked)))))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kart/validate! (assoc artifact :arguments (vec (reverse (:arguments artifact))))))))))

(deftest quotient-remainder-requires-the-same-checked-operands
  (let [original (quotient-neighbour-operation)]
    (doseq [replacement ['(- (long flat) (* q other-divisor))
                         '(- other-numerator (* q divisor))]]
      (let [operation (-> original
                          (update :scalars into #{'other-divisor 'other-numerator})
                          (assoc-in [:scalar-region :locals 1 :init] replacement))
            result (try
                     (sg/generate-scheduled-segmap-kernel
                      operation
                      :scalar-types {'n :long 'divisor :long 'width :long
                                     'other-divisor :long 'other-numerator :long}
                      :array-types {'fine :double 'output :double}
                      :target-dialect :opencl-portable)
                     nil
                     (catch clojure.lang.ExceptionInfo error (ex-data error)))]
        (is (= :kernel-body-index-precondition (:reason result))
            (str "unrelated Long operands cannot claim quotient correlation: " replacement))))))

(deftest actual-keyword-operand-cannot-overwrite-conditional-index-prefix
  (let [operation (assoc (quotient-neighbour-operation)
                         :scalar-region
                         {:locals [{:id 'q :dtype :long :init '(quot (long flat) divisor)}
                                   {:id 'one :dtype :long :init '(+ q (long 1))}]
                          :result '(clojure.core/aget fine (+ q width one))})
        scheduled (segmap-body/schedule
                   operation {:scalar-types {'n :long 'divisor :long 'width :long}
                              :array-types {'fine :double 'output :double}})
        checked (:body scheduled)
        computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                         (tree-seq coll? seq checked))
        addition (first (filter #(= :+ (get-in % [:expression :op])) computes))
        id (get-in addition [:result :id])
        keyword-id :raster.compiler.ir.kernel-body/index-prefix
        renamed (walk/postwalk
                 (fn [value]
                   (cond
                     (= id value) keyword-id
                     (and (instance? raster.compiler.ir.kernel_body.IndexCast value)
                          (= keyword-id (:argument value))
                          (= :long (:dtype value)) (= :exact (:overflow value)))
                     (:argument value)
                     :else value)) checked)
        original kernel-body/conditional-index-info
        captured (atom nil)]
    (is (= checked (kernel-body/validate! checked)) "original checked producer is valid")
    (is (some? id) "actual typed q+1 producer exists")
    (with-redefs [kernel-body/conditional-index-info
                  (fn [expression values]
                    (when (and (instance? raster.compiler.ir.kernel_body.IndexExpr expression)
                               (= 3 (count (:arguments expression)))
                               (get values keyword-id))
                      (reset! captured [expression values]))
                    (original expression values))]
      (try (kernel-body/validate! renamed)
           (catch clojure.lang.ExceptionInfo error
             (is (= :kernel-body-index-precondition (:reason (ex-data error)))))))
    (is (some? @captured) "actual verifier facts at the load include the legal keyword SSA")
    (let [[coordinate values] @captured
          info (original coordinate values)
          conditions (:required-preconditions info)
          candidate (assoc-in renamed [:attributes :required-preconditions] conditions)
          bindings (fn [n divisor width]
                     (fn [id] (case id _n_bound n divisor divisor width width)))]
      (is (some? info))
      (is (= candidate (kernel-body/validate! candidate)))
      (doseq [width [0 1 17]]
        (is (precondition/check! conditions (bindings 1 1 width))))
      (is (precondition/check! conditions (bindings 0 Long/MIN_VALUE Long/MAX_VALUE)))
      (is (thrown? ArithmeticException (Math/addExact Long/MAX_VALUE (long 1))))
      (is (thrown? ArithmeticException
                   (precondition/check! conditions (bindings 1 1 Long/MAX_VALUE)))
          "a legal real keyword identity cannot overwrite the actual overflowing prefix bound"))))

(deftest actual-direct-remainder-neighbour-chain-emits-with-closed-contracts
  (let [operation (assoc-in (quotient-neighbour-operation)
                            [:scalar-region :locals 1 :init] '(rem (long flat) divisor))
        options {:scalar-types {'n :long 'divisor :long 'width :long}
                 :array-types {'fine :double 'output :double}}
        scheduled (segmap-body/schedule operation options)
        result (try
                 (sg/generate-scheduled-segmap-kernel
                  operation :scalar-types (:scalar-types options)
                  :array-types (:array-types options) :target-dialect :opencl-portable)
                 nil
                 (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled)))
        "the actual direct rem topology is checked before the doubled remainder")
    (is (nil? result) (str "complete direct-rem neighbour emission: " result))))

(deftest guarded-positive-long-decrement-emits-without-global-restriction
  (let [operation (segop/->SegMap
                   906 (segop/make-seg-space 'row 'n) (segop/->SegLevel :thread :virtual)
                   '(if (> left (long 0)) (double (- left (long 1))) 0.0)
                   nil #{} #{'output} #{'left} (segop/->KernelGrid 1 256 0)
                   :double 'output nil)
        options {:scalar-types {'n :long 'left :long} :array-types {'output :double}}
        scheduled (segmap-body/schedule operation options)
        result (try
                 (sg/generate-scheduled-segmap-kernel
                  operation :scalar-types (:scalar-types options)
                  :array-types (:array-types options) :target-dialect :opencl-portable)
                 nil
                 (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled)))
        "the valid actual left>0 branch dominates left-1")
    (is (nil? result) (str "branch-local Long decrement emission: " result))))

(deftest guarded-uniform-long-product-emits-with-actual-abi-contract
  (let [operation (segop/->SegMap
                   907 (segop/make-seg-space 'hq 'n) (segop/->SegLevel :thread :virtual)
                   '(double (* n-kv head-dim)) nil #{} #{'output} #{'n-kv 'head-dim}
                   (segop/->KernelGrid 1 256 0) :double 'output nil)
        options {:scalar-types {'n :long 'n-kv :long 'head-dim :long}
                 :array-types {'output :double}}
        scheduled (segmap-body/schedule operation options)
        result (try
                 (sg/generate-scheduled-segmap-kernel
                  operation :scalar-types (:scalar-types options)
                  :array-types (:array-types options) :target-dialect :opencl-portable)
                 nil
                 (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled)))
        "the original attention product uses two actual Long ABI identities")
    (is (nil? result) (str "active-domain uniform ABI product emission: " result))))

(defn- remaining-product-map [form captures]
  (segop/->SegMap 908 (segop/make-seg-space 'row 'n)
                  (segop/->SegLevel :thread :virtual) form nil #{} #{'output}
                  captures (segop/->KernelGrid 1 256 0) :double 'output nil))

(defn- remaining-product-artifact [operation scalar-types]
  (sg/generate-scheduled-segmap-kernel operation :scalar-types scalar-types
                                       :array-types {'output :double 'fine :double}
                                       :target-dialect :opencl-portable))

(defn- remaining-product-trap [operation scalar-types]
  (let [scheduled (segmap-body/schedule operation {:scalar-types scalar-types
                                                  :array-types {'output :double 'fine :double}})]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled)))
        "negative controls reach checked typed IR before emission")
    (try (remaining-product-artifact operation scalar-types) nil
         (catch clojure.lang.ExceptionInfo error (select-keys (ex-data error) [:reason :operation])))))

(deftest positive-comparison-ranges-are-strictly-branch-local
  (let [types {'n :long 'left :long}
        artifact (remaining-product-artifact
                  (remaining-product-map '(if (> left (long 0))
                                            (double (- left (long 1))) 0.0) #{'left}) types)
        required (:preconditions artifact)]
    (doseq [left [Long/MIN_VALUE -1 0 1 Long/MAX_VALUE]]
      (is (precondition/check! required (fn [id] (case id _n_bound 1 left left)))
          "negative left remains legal in the unchanged else arm"))
    (is (not-any? #(= 'left (:expression %)) required)
        "the comparison supplies lexical range facts, not a host left restriction")
    (doseq [form ['(if (> left (long 0)) 0.0 (double (- left (long 1))))
                  '(if (< left (long 0)) (double (- left (long 1))) 0.0)
                  '(let [x (- left (long 1))] (if (> left (long 0)) (double x) 0.0))
                  '(+ (if (> left (long 0)) (double (- left (long 1))) 0.0)
                      (double (- left (long 1))))
                  '(if (> other (long 0)) (double (- left (long 1))) 0.0)]]
      (is (= {:reason :kernel-body-c-trap-unsupported :operation :-}
             (remaining-product-trap
              (cond-> (remaining-product-map form #{'left 'other})
                (= 'let (first form))
                (assoc :scalar-region {:locals [{:id 'x :dtype :long :init (second (second form))}]
                                       :result (nth form 2)}))
              (assoc types 'other :long)))))))

(deftest uniform-product-contracts-use-the-actual-active-domain
  (let [artifact (remaining-product-artifact
                  (remaining-product-map '(double (* n-kv head-dim)) #{'n-kv 'head-dim})
                  {'n :long 'n-kv :long 'head-dim :long})
        required (:preconditions artifact)
        bindings (fn [n a b] (fn [id] (case id _n_bound n n-kv a head-dim b)))]
    (doseq [a [Long/MIN_VALUE -1 Long/MAX_VALUE]
            b [Long/MIN_VALUE -1 Long/MAX_VALUE]]
      (is (precondition/check! required (bindings 0 a b))))
    (is (precondition/check! required (bindings 1 Long/MAX_VALUE 1)))
    (is (precondition/check! required (bindings 1 2147483648 2)))
    (doseq [[a b] [[Long/MAX_VALUE 2] [Long/MIN_VALUE -1] [-1 1]]]
      (is (thrown? Exception (precondition/check! required (bindings 1 a b)))
          "this closed cohort rejects negative factors and actual overflow"))
    (is (= {:reason :kernel-body-c-trap-unsupported :operation :-}
           (remaining-product-trap
            (remaining-product-map '(double (- total (* n-kv head-dim)))
                                   #{'total 'n-kv 'head-dim})
            {'n :long 'total :long 'n-kv :long 'head-dim :long}))
        "product representability cannot certify a separate possibly underflowing subtraction")))

(deftest direct-remainder-retains-divisor-and-address-obligations
  (let [operation (assoc-in (quotient-neighbour-operation)
                            [:scalar-region :locals 1 :init] '(rem (long flat) divisor))
        types {'n :long 'divisor :long 'width :long}
        artifact (remaining-product-artifact operation types)
        required (:preconditions artifact)
        bindings (fn [n d w] (fn [id] (case id _n_bound n divisor d width w)))]
    (doseq [d [Long/MIN_VALUE 0 Long/MAX_VALUE]]
      (is (precondition/check! required (bindings 0 d Long/MAX_VALUE))))
    (doseq [d [0 -1]]
      (is (thrown? Exception (precondition/check! required (bindings 1 d 0)))))
    (is (precondition/check! required (bindings 1 Long/MAX_VALUE 0)))
    (is (thrown? ArithmeticException
                 (precondition/check! required (bindings 1 Long/MAX_VALUE Long/MAX_VALUE))))
    (doseq [init ['(rem other divisor)]]
      (is (= :kernel-body-index-precondition
             (try (remaining-product-artifact
                   (-> operation (assoc-in [:scalar-region :locals 1 :init] init)
                       (update :scalars conj 'other)) (assoc types 'other :long)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "an unrelated numerator cannot inherit the guarded index's range"))
    (let [separate (remaining-product-artifact
                    (-> operation (assoc-in [:scalar-region :locals 1 :init] '(rem (long flat) other))
                        (update :scalars conj 'other)) (assoc types 'other :long))]
      (is (= separate (kart/validate! separate)))
      (is (precondition/check! (:preconditions separate)
                              (fn [id] (case id _n_bound 1 divisor 1 other 1 width 0))))
      (is (thrown? Exception
                   (precondition/check! (:preconditions separate)
                                        (fn [id] (case id _n_bound 1 divisor 1 other 0 width 0))))
          "general checked addition retains the distinct remainder divisor obligation"))))


(deftest remaining-product-proofs-are-reconstructed-at-the-actual-guard
  (doseq [[operation types]
          [[(remaining-product-map '(double (* n-kv head-dim)) #{'n-kv 'head-dim})
            {'n :long 'n-kv :long 'head-dim :long}]
           [(assoc-in (quotient-neighbour-operation)
                      [:scalar-region :locals 1 :init] '(rem (long flat) divisor))
            {'n :long 'divisor :long 'width :long}]]]
    (let [checked (:body (segmap-body/schedule operation {:scalar-types types
                                                        :array-types {'output :double 'fine :double}}))
          unguarded (update checked :operations
                            (fn [operations]
                              (vec (mapcat (fn [operation]
                                             (if (instance? raster.compiler.ir.kernel_body.Guard operation)
                                               (:operations operation) [operation])) operations))))
          required (kernel-body/required-preconditions checked)]
      (is (= checked (kernel-body/validate! checked)))
      (is (seq required))
      (is (= :kernel-body-intrinsic-overflow
             (try (kernel-body/validate! unguarded) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "schema-valid operations outside the guard cannot borrow lexical range authority")
      (doseq [condition required]
        (is (thrown? clojure.lang.ExceptionInfo
                     (kernel-body/validate!
                      (assoc-in checked [:attributes :required-preconditions]
                                (vec (remove #{condition} required)))))
            "the standalone validator reconstructs each intermediate obligation")))))

(deftest explicit-store-recomposes-the-checked-quotient-and-direct-remainder
  (let [operation (assoc (remaining-product-map
                          '(clojure.core/aset output (+ (* q divisor) r) 0.0) #{'divisor})
                         :out-sym nil
                         :scalar-region {:locals [{:id 'q :dtype :long :init '(quot (long row) divisor)}
                                                  {:id 'r :dtype :long :init '(rem (long row) divisor)}]
                                         :result '(clojure.core/aset output (+ (* q divisor) r) 0.0)})
        original kernel-body/validate!
        candidate (atom nil)
        result (with-redefs [kernel-body/validate!
                            (fn [& args]
                              (when (some #(instance? raster.compiler.ir.kernel_body.ScalarStore %)
                                          (tree-seq coll? seq (first args)))
                                (reset! candidate (first args)))
                              (apply original args))]
                 (try {:artifact (remaining-product-artifact operation {'n :long 'divisor :long})}
                      (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))
        stores (filter #(instance? raster.compiler.ir.kernel_body.ScalarStore %)
                       (tree-seq coll? seq @candidate))
        computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                         (tree-seq coll? seq @candidate))]
    (is (= 1 (count stores)) "the normal route retains the actual explicit destination Store")
    (is (= #{:quot :rem} (set (keep #(when (contains? #{:quot :rem} (get-in % [:expression :op]))
                                      (get-in % [:expression :op])) computes)))
        "both actual typed source definitions precede the Store")
    (is (nil? (:error result)) (str "the correlated Store must validate and emit: " (:error result)))))

(deftest explicit-store-addresses-collect-nonrecomposition-obligations
  (let [operation (-> (quotient-neighbour-operation)
                      (assoc :out-sym nil)
                      (assoc-in [:scalar-region :locals 1 :init] '(rem (long flat) divisor))
                      (assoc-in [:scalar-region :result]
                                '(clojure.core/aset output (+ k width 1) 0.0)))
        artifact (remaining-product-artifact operation {'n :long 'divisor :long 'width :long})
        bindings (fn [n d w] (fn [id] (case id _n_bound n divisor d width w)))]
    (is (= artifact (kart/validate! artifact)))
    (is (precondition/check! (:preconditions artifact) (bindings 0 Long/MIN_VALUE Long/MAX_VALUE)))
    (is (precondition/check! (:preconditions artifact) (bindings 1 Long/MAX_VALUE 0)))
    (is (thrown? ArithmeticException
                 (precondition/check! (:preconditions artifact)
                                      (bindings 1 Long/MAX_VALUE Long/MAX_VALUE)))
        "the actual destination +width+1 obligation rejects active overflow")))

(defn- remainder-distance-fold-operation []
  (assoc (remaining-product-map nil #{'divisor})
         :scalar-region
         {:locals [{:id 'r :dtype :long :init '(rem (long row) divisor)}]
          :result '(fold {:accumulator acc :index i :identity 0.0 :lower 0
                          :extent divisor :dtype :double :association :ordered}
                         (lambda [acc i] (region [] [(+ acc (double (- r i)))])))}))

(defn- uniform-stride-effect-loop-operation []
  (assoc (remaining-product-map nil #{'cache-len 'n-kv 'head-dim}) :out-sym nil
         :scalar-region
         {:locals [{:id 'stride :dtype :long :init '(* n-kv head-dim)}]
          :effects [{:loop {:index 't :lower 0 :extent 'cache-len :upper-bound :exclusive
                            :carries []
                            :locals [{:id 'offset :dtype :long :init '(* t stride)}]
                            :effects [{:destination 'output :destination-index 'offset
                                       :value 0.0 :predicate 1 :conflict {:kind :unique}}]}}]}))

(deftest canonical-long-fold-distance-retains-entered-body-bounds
  (let [operation (remainder-distance-fold-operation)
        types {'n :long 'divisor :long}
        scheduled (segmap-body/schedule operation {:scalar-types types :array-types {'output :double}})
        loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                      (tree-seq coll? seq (:body scheduled)))
        result (try (remaining-product-artifact operation types) nil
                    (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (= [[:long 1 :ordered]]
           (mapv #(vector (get-in % [:index :type]) (:step %) (get-in % [:attributes :association])) loops))
        "the actual ordered Fold is a canonical Long exclusive step-one ForLoop")
    (is (nil? result) (str "the signed remainder-minus-entered-index distance must emit: " result))))

(deftest canonical-long-effect-loop-multiplies-the-checked-uniform-stride
  (let [operation (uniform-stride-effect-loop-operation)
        types {'n :long 'cache-len :long 'n-kv :long 'head-dim :long}
        scheduled (segmap-body/schedule operation {:scalar-types types :array-types {'output :double}})
        loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                      (tree-seq coll? seq (:body scheduled)))
        result (try (remaining-product-artifact operation types) nil
                    (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (= [[:long 1 :ordered]]
           (mapv #(vector (get-in % [:index :type]) (:step %) (get-in % [:attributes :association])) loops))
        "the actual effect local multiplies a real canonical entered loop index")
    (is (nil? result) (str "loop-index times checked ABI stride must emit: " result))))

(defn- complete-attention-address-operation []
  ;; Retained Long source metadata is the checked IR vocabulary, not a new storage conversion.
  (let [long-expression (fn [form] (walk/postwalk #(if (seq? %) (with-meta % {:raster.type/tag 'long}) %) form))]
    (assoc (remaining-product-map nil #{'cache-len 'n-kv 'head-dim 'group})
           :space (segop/make-seg-space 'hq 'n) :out-sym nil :dtype :float
           :scalar-region
           {:locals [{:id 'stride :dtype :long :init (long-expression '(* n-kv head-dim))}
                     {:id 'head-offset :dtype :long :init (long-expression '(* (quot (long hq) group) head-dim))}]
            :effects [{:loop {:index 't :lower 0 :extent 'cache-len :upper-bound :exclusive
                              :carries []
                              :locals [{:id 'base :dtype :long :init (long-expression '(+ (* t stride) head-offset))}]
                              :effects [{:loop {:index 'j :lower 0 :extent 'head-dim :upper-bound :exclusive
                                                :carries []
                                                :locals [{:id 'address :dtype :long :init (long-expression '(+ base j))}]
                                                :effects [{:destination 'output :destination-index 'address
                                                           :value '(float 0.0) :predicate 1
                                                           :conflict {:kind :unique}}]}}]}}]})))

(deftest complete-attention-address-chain-retains-both-loop-domains
  (let [operation (complete-attention-address-operation)
        types {'n :long 'cache-len :long 'n-kv :long 'head-dim :long 'group :long}
        scheduled (segmap-body/schedule operation {:scalar-types types :array-types {'output :float}})
        loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                      (tree-seq coll? seq (:body scheduled)))
        stores (filter #(instance? raster.compiler.ir.kernel_body.ScalarStore %)
                       (tree-seq coll? seq (:body scheduled)))
        result (try (sg/generate-scheduled-segmap-kernel operation :scalar-types types
                                                        :array-types {'output :float}
                                                        :target-dialect :opencl-portable)
                    nil (catch clojure.lang.ExceptionInfo error (ex-data error)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (= 2 (count loops)) "outer cache and inner head-dimension loops both remain")
    (is (= [:long :long] (mapv #(get-in % [:index :type]) loops)))
    (is (= 1 (count stores)) "the complete address reaches its actual Float destination Store")
    (is (nil? result) (str "the complete checked attention address chain must emit: " result))))

(defn- recomposition-store-operation [remainder-form captures]
  (assoc (remaining-product-map nil captures) :out-sym nil
         :scalar-region
         {:locals [{:id 'q :dtype :long :init '(quot (long row) divisor)}
                   {:id 'r :dtype :long :init remainder-form}]
          :result '(clojure.core/aset output (+ (* q divisor) r) 0.0)}))

(deftest explicit-store-recomposition-keeps-actual-divisor-and-numerator-identities
  (let [types {'n :long 'divisor :long 'other :long}
        bindings (fn [n d other] (fn [id] (case id _n_bound n divisor d other other)))
        same (remaining-product-artifact
              (recomposition-store-operation '(rem (long row) divisor) #{'divisor}) types)]
    (is (= same (kart/validate! same)))
    (is (precondition/check! (:preconditions same) (bindings Long/MAX_VALUE 1 1)))
    (doseq [remainder ['(rem (long row) other) '(rem other divisor)]]
      (is (= :kernel-body-index-precondition
             (try (remaining-product-artifact
                   (recomposition-store-operation remainder #{'divisor 'other}) types) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "distinct actual divisor/numerator identities cannot fall through unresolved address arithmetic"))))

(deftest explicit-store-and-atomic-addresses-retain-independent-obligations
  (doseq [store ['(clojure.core/aset output (+ k width 1) 0.0)
                 '(raster.par/atomic-add! output (+ k width 1) 0.0)]]
    (let [operation (-> (quotient-neighbour-operation)
                        (assoc :out-sym nil)
                        (assoc-in [:scalar-region :locals 1 :init] '(rem (long flat) divisor))
                        (assoc-in [:scalar-region :result] store))
          types {'n :long 'divisor :long 'width :long}
          operation (cond-> operation
                      (= 'raster.par/atomic-add! (first store))
                      (assoc :write-conflicts {'output {:kind :reduce :operator :+}}
                             :write-conflict :reduce))
          scheduled (segmap-body/schedule operation {:scalar-types types
                                                    :array-types {'output :double 'fine :double}})
          checked (:body scheduled)
          required (kernel-body/required-preconditions checked)
          bindings (fn [n d w] (fn [id] (case id _n_bound n divisor d width w)))
          destination (filter #(or (instance? raster.compiler.ir.kernel_body.ScalarStore %)
                                   (instance? raster.compiler.ir.kernel_body.AtomicRMW %))
                              (tree-seq coll? seq checked))]
      (is (= checked (kernel-body/validate! checked)))
      (is (= 1 (count destination)))
      (is (precondition/check! required (bindings 0 Long/MIN_VALUE Long/MAX_VALUE)))
      (is (thrown? ArithmeticException
                   (precondition/check! required (bindings 1 Long/MAX_VALUE Long/MAX_VALUE))))
      (doseq [condition required]
        (is (thrown? clojure.lang.ExceptionInfo
                     (kernel-body/validate!
                      (assoc-in checked [:attributes :required-preconditions]
                                (vec (remove #{condition} required)))))
            "the standalone validator reconstructs each destination obligation")))))

(deftest explicit-recomposition-store-requires-its-dominating-active-domain
  (let [operation (recomposition-store-operation '(rem (long row) divisor) #{'divisor})
        checked (:body (segmap-body/schedule operation {:scalar-types {'n :long 'divisor :long}
                                                       :array-types {'output :double}}))
        unguarded (update checked :operations
                          (fn [operations]
                            (vec (mapcat (fn [operation]
                                           (if (instance? raster.compiler.ir.kernel_body.Guard operation)
                                             (:operations operation) [operation])) operations))))
        result (try (kernel-body/validate! unguarded) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))]
    (is (= checked (kernel-body/validate! checked)))
    (is (contains? #{:kernel-body-intrinsic-overflow :kernel-body-index-precondition} result)
        (str "schema-valid Store arithmetic outside the active guard cannot borrow its quotient domain: " result))))

(deftest canonical-loop-facts-require-the-actual-exclusive-zero-step-one-form
  (let [operation (uniform-stride-effect-loop-operation)
        types {'n :long 'cache-len :long 'n-kv :long 'head-dim :long}
        checked (:body (segmap-body/schedule operation {:scalar-types types
                                                       :array-types {'output :double}}))
        loop-operation (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                                      (tree-seq coll? seq checked)))
        upper (:upper loop-operation)
        upper-info {:type :long :range {:lower Long/MIN_VALUE :upper Long/MAX_VALUE}
                    :abi-scalar-identity 'cache-len}]
    (is (= checked (kernel-body/validate! checked)))
    (is (= 'cache-len (if (instance? raster.compiler.ir.kernel_body.IndexCast upper)
                        (:argument upper) upper)))
    (is (= 1 (:step loop-operation)))
    (is (not= :inclusive (get-in loop-operation [:attributes :upper-bound])))
    (is (some? (kernel-body/entered-for-loop-info (:lower loop-operation) upper-info 1 :exclusive)))
    (doseq [[lower step mode] [[1 1 :exclusive] [0 2 :exclusive] [0 1 :inclusive]]]
      (is (nil? (kernel-body/entered-for-loop-info lower upper-info step mode))
          "noncanonical loops cannot acquire the entered-zero-origin proof"))))

(deftest empty-inner-loops-preserve-evaluated-prefix-contracts
  (let [artifact (remaining-product-artifact
                  (uniform-stride-effect-loop-operation)
                  {'n :long 'cache-len :long 'n-kv :long 'head-dim :long})
        required (:preconditions artifact)
        bindings (fn [n cache a b]
                   (fn [id] (case id _n_bound n cache-len cache n-kv a head-dim b)))]
    (doseq [cache [Long/MIN_VALUE 0 Long/MAX_VALUE]]
      (is (precondition/check! required (bindings 0 cache Long/MIN_VALUE Long/MAX_VALUE))))
    (doseq [cache [Long/MIN_VALUE 0 1 2]]
      (is (precondition/check! required (bindings 1 cache Long/MAX_VALUE 1))))
    (doseq [cache [Long/MIN_VALUE 0]]
      (is (thrown? ArithmeticException
                   (precondition/check! required (bindings 1 cache Long/MAX_VALUE 2)))
          "the stride executes before the empty inner loop"))
    (is (thrown? ArithmeticException
                 (precondition/check! required (bindings 1 3 Long/MAX_VALUE 1))))))

(deftest complete-attention-sums-check-combined-bounds-and-evaluation-order
  (let [artifact (sg/generate-scheduled-segmap-kernel
                  (complete-attention-address-operation)
                  :scalar-types {'n :long 'cache-len :long 'n-kv :long 'head-dim :long 'group :long}
                  :array-types {'output :float} :target-dialect :opencl-portable)
        required (:preconditions artifact)
        bindings (fn [n cache kv dim group]
                   (fn [id] (case id _n_bound n cache-len cache n-kv kv head-dim dim group group)))]
    (is (= artifact (kart/validate! artifact)))
    (is (precondition/check! required (bindings 0 Long/MAX_VALUE Long/MIN_VALUE Long/MAX_VALUE Long/MIN_VALUE)))
    (is (precondition/check! required (bindings 2 1 1 1 1)))
    (is (thrown? ArithmeticException
                 (precondition/check! required (bindings 2 2 1 Long/MAX_VALUE 1)))
        "each head/loop product fits while their actual sum overflows")
    (is (thrown? ArithmeticException
                 (precondition/check! required (bindings 1 0 Long/MAX_VALUE 2 1)))
        "an unentered cache loop cannot erase the already evaluated stride product")))

(deftest canonical-loop-authority-is-reconstructed-from-the-actual-for-loop
  (let [operation (uniform-stride-effect-loop-operation)
        types {'n :long 'cache-len :long 'n-kv :long 'head-dim :long}
        checked (:body (segmap-body/schedule operation {:scalar-types types
                                                       :array-types {'output :double}}))
        checked (update checked :parameters conj
                        (assoc (first (filter #(= 'cache-len (:id %)) (:parameters checked))) :id 'other))
        mutations [(fn [loop-operation]
                     (assoc loop-operation :lower (kernel-body/->IndexCast 'other :long :exact)))
                   (fn [loop-operation] (assoc loop-operation :step 2))
                   (fn [loop-operation] (assoc-in loop-operation [:attributes :upper-bound] :inclusive))]]
    (is (= checked (kernel-body/validate! checked)))
    (doseq [mutate mutations]
      (let [candidate (walk/postwalk
                       #(if (instance? raster.compiler.ir.kernel_body.ForLoop %) (mutate %) %)
                       checked)
            result (try (kernel-body/validate! candidate) nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))]
        (is (= :kernel-body-intrinsic-overflow result)
            (str "a schema-valid noncanonical loop cannot retain the old multiplication proof: " result))))))

(deftest quotient-address-ancestry-survives-checked-aliases-and-backedges
  (let [operation (recomposition-store-operation '(rem (long row) divisor) #{'divisor})
        checked (:body (segmap-body/schedule operation {:scalar-types {'n :long 'divisor :long}
                                                       :array-types {'output :double}}))
        q (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                               (= :quot (get-in % [:expression :op]))) (tree-seq coll? seq checked)))
        q-id (get-in q [:result :id])
        with-alias (walk/postwalk
                    (fn [node]
                      (if (instance? raster.compiler.ir.kernel_body.Guard node)
                        (update node :operations
                                #(vec (mapcat (fn [op]
                                                (if (= op q)
                                                  [op (kernel-body/->ScalarCompute
                                                       (kernel-body/value 'checked-q-alias :long)
                                                       (kernel-body/scalar-expression :cast :long [q-id]
                                                                                      {:rounding :exact :overflow :exact}))
                                                   (kernel-body/->IndexCompute 'checked-q-index
                                                                               (kernel-body/->IndexCast 'checked-q-alias :long :exact))]
                                                  [(if (instance? raster.compiler.ir.kernel_body.ScalarStore op)
                                                     (update op :coordinates
                                                             (partial walk/postwalk-replace {q-id 'checked-q-index})) op)])) %))) node)) checked)
        unguarded (update with-alias :operations
                          #(vec (mapcat (fn [op]
                                          (if (instance? raster.compiler.ir.kernel_body.Guard op)
                                            (:operations op) [op])) %)))
        loop-body (fn [upper carry-result coordinate]
                    (kernel-body/->ForLoop
                     (kernel-body/value 'ancestry-loop-index :long)
                     (kernel-body/->IndexCast 0 :long :exact)
                     (kernel-body/->IndexCast upper :long :exact) 1
                     [(kernel-body/->LoopArg (kernel-body/value 'first-carry :long)
                                             (kernel-body/literal 0 :long))
                      (kernel-body/->LoopArg (kernel-body/value 'second-carry :long)
                                             (kernel-body/literal 0 :long))]
                     [(kernel-body/->ScalarStore 'output [coordinate]
                                                (kernel-body/literal 0.0 :double) :map-active)
                      (kernel-body/->Yield [carry-result 'first-carry])]
                     [(kernel-body/value 'first-loop-result :long)
                      (kernel-body/value 'second-loop-result :long)]
                     {:association :ordered :source-order true}))
        replace-store (fn [replacement]
                        (walk/postwalk #(if (instance? raster.compiler.ir.kernel_body.ScalarStore %) replacement %)
                                       checked))
        ordinary (replace-store (loop-body 3 'first-carry 'second-carry))
        later (replace-store (loop-body 3 q-id
                                       (kernel-body/->IndexExpr :mul ['second-carry
                                                                    (kernel-body/->IndexCast 'divisor :long :exact)])))]
    (is (= checked (kernel-body/validate! checked)))
    (is (= with-alias (kernel-body/validate! with-alias))
        "exact ScalarExpr and IndexCompute aliases preserve the checked quotient relation")
    (is (= :kernel-body-index-precondition
           (try (kernel-body/validate! unguarded) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= ordinary (kernel-body/validate! ordinary))
        "ordinary no-quotient carries retain the existing checked address route")
    (is (= :kernel-body-index-precondition
           (try (kernel-body/validate! later) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "a second carry becomes quotient-derived only on a later backedge and cannot escape closure")
    (let [zero-trip (replace-store (loop-body 0 q-id 'second-carry))]
      (is (= zero-trip (kernel-body/validate! zero-trip))
          "zero-trip carry results still retain their initial type/storage contracts"))))


(deftest signed-donor-coordinate-with-unit-remainder-has-an-ordinary-proof
  (let [operation (assoc (remaining-product-map nil #{'i})
                         :inputs #{'donors 'fine}
                         :scalar-region
                         {:locals [{:id 'donor :dtype :int :init '(aget donors row)}
                                   {:id 'r :dtype :long :init '(rem i (long 1))}]
                          :result '(aget fine (+ (* (long donor) (long 1)) r))})
        result (try
                 {:artifact (sg/generate-scheduled-segmap-kernel
                             operation :scalar-types {'n :long 'i :long}
                             :array-types {'donors :int 'fine :double 'output :double}
                             :target-dialect :opencl-portable)}
                 (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
    (is (nil? (:error result))
        (str "signed Int32 donor plus rem(signed Long,1) must retain the ordinary address proof: " (:error result)))))

(defn- nested-causal-attention-address-operation []
  (let [long-expression (fn [form] (walk/postwalk #(if (seq? %) (with-meta % {:raster.type/tag 'long}) %) form))]
    (assoc (remaining-product-map nil #{'ss 'seq-len 'slab 'head-dim})
           :space (segop/make-seg-space 't 'n) :inputs #{'query 'key}
           :scalar-region
           {:locals [{:id 'q1 :dtype :long :init (long-expression '(quot (long t) ss))}
                     {:id 'r1 :dtype :long :init (long-expression '(rem (long t) ss))}
                     {:id 'q2 :dtype :long :init (long-expression '(quot r1 seq-len))}
                     {:id 'r2 :dtype :long :init (long-expression '(rem r1 seq-len))}
                     {:id 'batch :dtype :long :init (long-expression '(* q1 slab))}
                     {:id 'qbase :dtype :long :init (long-expression '(+ batch (* q2 head-dim)))}
                     {:id 'kbase :dtype :long :init (long-expression '(+ batch (* r2 head-dim)))}]
            :result '(if (<= r2 q2)
                       (fold {:accumulator acc :index j :identity 0.0 :lower 0
                              :extent head-dim :dtype :double :association :ordered}
                             (lambda [acc j]
                               (region [] [(+ acc (* (aget query (+ qbase j))
                                                    (aget key (+ kbase j))))])))
                       0.0)})))

(deftest nested-causal-attention-address-retains-the-actual-nested-numerator
  (let [operation (nested-causal-attention-address-operation)
        result (try
                 {:artifact (sg/generate-scheduled-segmap-kernel
                             operation :scalar-types {'n :long 'ss :long 'seq-len :long 'slab :long 'head-dim :long}
                             :array-types {'query :double 'key :double 'output :double}
                             :target-dialect :opencl-portable)}
                 (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
    (is (nil? (:error result))
        (str "the complete nested causal q/r, products, ordered loop and load coordinates must emit: " (:error result)))))


(deftest nested-causal-address-contract-retains-evaluated-prefix-and-empty-activation
  (let [operation (nested-causal-attention-address-operation)
        types {'n :long 'ss :long 'seq-len :long 'slab :long 'head-dim :long}
        options {:scalar-types types :array-types {'query :double 'key :double 'output :double}}
        scheduled (segmap-body/schedule operation options)
        artifact (sg/generate-scheduled-segmap-kernel operation
                   :scalar-types types :array-types (:array-types options) :target-dialect :opencl-portable)
        loads (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                      (tree-seq coll? seq (:body scheduled)))
        computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                         (tree-seq coll? seq (:body scheduled)))
        bindings (fn [n ss seq-len slab hd]
                   (fn [id] (case id _n_bound n ss ss seq-len seq-len slab slab head-dim hd)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (= #{'query 'key} (set (map :buffer loads))))
    (is (= 4 (count (filter #(contains? #{:quot :rem} (get-in % [:expression :op])) computes))))
    (is (precondition/check! (:preconditions artifact)
                            (bindings 0 Long/MIN_VALUE Long/MIN_VALUE Long/MAX_VALUE Long/MAX_VALUE)))
    (is (precondition/check! (:preconditions artifact) (bindings 4 4 2 8 2)))
    (doseq [tuple [[1 1 0 0 0] [1 1 -1 0 0] [4 4 2 8 Long/MAX_VALUE]]]
      (is (thrown? Exception
                   (precondition/check! (:preconditions artifact) (apply bindings tuple)))
          "the actual nested divisor and all evaluated products remain checked even before an empty inner loop"))))


(deftest nested-causal-raw-ordered-loop-collects-the-final-load-obligation
  (let [operation (assoc-in (nested-causal-attention-address-operation)
                            [:scalar-region :result]
                            '(if (<= r2 q2)
                               (loop [j 0 acc 0.0]
                                 (if (< j head-dim)
                                   (recur (inc j) (+ acc (* (aget query (+ qbase j))
                                                           (aget key (+ kbase j)))))
                                   acc))
                               0.0))
        types {'n :long 'ss :long 'seq-len :long 'slab :long 'head-dim :long}
        scheduled (segmap-body/schedule operation {:scalar-types types
                                                  :array-types {'query :double 'key :double 'output :double}})
        loads (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                      (tree-seq coll? seq (:body scheduled)))
        artifact (sg/generate-scheduled-segmap-kernel operation :scalar-types types
                   :array-types {'query :double 'key :double 'output :double}
                   :target-dialect :opencl-portable)]
    (is (= #{'query 'key} (set (map :buffer loads))))
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (seq (:preconditions artifact)))
    (is (thrown? Exception
                 (precondition/check! (:preconditions artifact)
                   (fn [id] (case id _n_bound 4 ss 4 seq-len 2 slab 8 head-dim Long/MAX_VALUE)))))))


(deftest nested-recomposition-keeps-inner-numerator-and-divisor-identities
  (let [base (-> (nested-causal-attention-address-operation)
                 (assoc :inputs #{'fine})
                 (update-in [:scalar-region :locals] #(vec (take 4 %)))
                 (assoc-in [:scalar-region :result] '(aget fine (+ (* q2 seq-len) r2))))
        types {'n :long 'ss :long 'seq-len :long 'slab :long 'head-dim :long 'other :long}
        positive (remaining-product-artifact base types)]
    (is (= positive (kart/validate! positive)))
    (doseq [r2 ['(rem r1 other) '(rem q1 seq-len)]]
      (let [different (-> base
                          (update :scalars conj 'other)
                          (assoc-in [:scalar-region :locals 3 :init] r2))]
        (is (= :kernel-body-index-precondition
               (try (remaining-product-artifact different types) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "an unrelated actual nested numerator/divisor cannot inherit the exact recomposition relation")))))


(deftest broad-valid-overflow-proof-does-not-erase-the-static-read-domain
  (let [graph (capacity-fixture/inferred-graph '(let* [k (+ i j)] (aget a k)))
        operation (get-in graph [:nodes 0 :operation])
        options {:dtype :float :array-types {'a :float 'C :float}}
        lowered (segmap-body/lower operation options)
        proofs (keep #(get-in % [:expression :options :proof])
                     (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                             (tree-seq coll? seq (:kernel-body lowered))))]
    (is (some #(and (= :typed-scalar-range (:kind %)) (< (:lower %) 0)) proofs)
        "the actual validated certificate conservatively includes negative values")
    (is (= {'a 6} (map-reads/static-read-requirements operation options)))
    (is (= {'a 6} (into {} (map (juxt :id :elements)) (:inputs graph))))
    (let [indirect (get-in (capacity-fixture/inferred-graph '(aget a (int (aget b j))))
                          [:nodes 0 :operation])]
      (is (nil? (map-reads/static-read-requirements
                 indirect {:dtype :float :array-types {'a :float 'b :float 'C :float}}))
          "unknown loaded contents cannot supply an input capacity"))))
