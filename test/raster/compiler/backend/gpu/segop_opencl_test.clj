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
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.scalar-range :as scalar-range]
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
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
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
    (is (= [{:expression '_n_bound :op :>= :value 0}] (:preconditions ordinary))
        "the actual nonnegative launch extent remains explicit even without address arithmetic")
    (is (true? (precondition/check! (:preconditions ordinary) (fn [_] 0))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (precondition/check! (:preconditions ordinary) (fn [_] -1))))
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
    (is (= (kernel-body/expression :floor-div 'scan-index
                                   (kernel-body/index-cast scan-width :long :exact))
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
    (is (= '[wide-group-index wide-group-count wide-local-index] (mapv :id casts)))
    (is (= (kernel-body/expression :max
                  (kernel-body/index-cast '_n_bound :long :exact)
                  (kernel-body/index-cast 0 :long :exact))
           (get index-expressions 'wide-bound))
        "only generated zero-trip geometry is clamped; the original bound conversion stays exact")
    (is (every? #(= :long (get-in % [:expression :dtype])) casts))
    (is (= :long (get-in loop [:index :type])))
    (is (= :ceil-div (get-in index-expressions ['group-chunk :op]))
        "the actual Long ceil-div addition prefix is independently checked, including zero")
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
        (is (= :kernel-body-c-trap-unsupported (:reason result))
            (str "unrelated Long operands retain checked arithmetic and portable emission rejects: " replacement))))))

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

(defn- assert-independent-address-contract [artifact]
  (let [checked (get-in artifact [:attributes :kernel-body])
        scheduled (get-in artifact [:attributes :scheduled-kernel-body])
        conditions (:preconditions artifact)]
    (is (= artifact (kart/validate! artifact)))
    (is (= conditions (kernel-body/required-preconditions checked)))
    (is (= conditions (:preconditions scheduled)))
    (is (= conditions (:required-preconditions
                        (scalar-emitter/emit-scalar-module
                         "independent_address_contract" checked {:target-dialect :opencl-portable}))))
    (is (some #(and (= :<= (:op %))
                    (instance? raster.compiler.ir.kernel_launch.Sum (:expression %))) conditions)
        "distinct operands retain a separately checked sum cap")
    (doseq [condition conditions]
      (is (contains? #{:kernel-body-index-precondition :kernel-body-intrinsic-overflow}
                     (try (kernel-body/validate!
                            (assoc-in checked [:attributes :required-preconditions]
                                      (vec (remove #{condition} conditions)))) :accepted
                          (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "every independent divisor/prefix condition is reconstructed at consumption"))))

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
      (is (= {:reason :kernel-body-c-trap-unsupported :operation :* :type :long}
             (try (remaining-product-artifact
                   (-> operation (assoc-in [:scalar-region :locals 1 :init] init)
                       (update :scalars conj 'other)) (assoc types 'other :long)) nil
                  (catch clojure.lang.ExceptionInfo error (select-keys (ex-data error) [:reason :operation :type]))))
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
              (recomposition-store-operation '(rem (long row) divisor) #{'divisor}) types)
        changed (remaining-product-artifact
                 (recomposition-store-operation '(rem (long row) other) #{'divisor 'other}) types)
        computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                         (tree-seq coll? seq (get-in changed [:attributes :kernel-body])))
        quotient (first (filter #(= :quot (get-in % [:expression :op])) computes))
        remainder (first (filter #(= :rem (get-in % [:expression :op])) computes))]
    (is (= same (kart/validate! same)))
    (is (precondition/check! (:preconditions same) (bindings Long/MAX_VALUE 1 1)))
    (is (= 'other (last (get-in remainder [:expression :arguments]))))
    (is (not= (get-in quotient [:expression :arguments])
              (get-in remainder [:expression :arguments])))
    (assert-independent-address-contract changed)
    (is (precondition/check! (:preconditions changed) (bindings 1 1 1)))
    (is (precondition/check! (:preconditions changed) (bindings 0 Long/MIN_VALUE Long/MIN_VALUE)))
    (doseq [d [0 -1]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (precondition/check! (:preconditions changed) (bindings 1 1 d)))))
    (is (thrown? ArithmeticException
                 (precondition/check! (:preconditions changed) (bindings Long/MAX_VALUE 1 3)))
        "the independently retained sum cap rejects MAX-1 plus a remainder bound of two")
    (is (= {:reason :kernel-body-c-trap-unsupported :operation :+ :type :long}
           (try (remaining-product-artifact
                  (recomposition-store-operation '(rem other divisor) #{'divisor 'other}) types) :accepted
                (catch clojure.lang.ExceptionInfo error
                  (select-keys (ex-data error) [:reason :operation :type])))))))

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
    (doseq [[position mutate] (map-indexed vector mutations)]
      (let [candidate (walk/postwalk
                       #(if (instance? raster.compiler.ir.kernel_body.ForLoop %) (mutate %) %)
                       checked)
            result (try (kernel-body/validate! candidate) nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))]
        (if (= position 1)
          (let [conditions (kernel-body/required-preconditions candidate)
                loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                              (tree-seq coll? seq candidate))]
            (is (= candidate (kernel-body/validate! candidate)))
            (is (every? #(= 2 (:step %)) loops))
            (is (= 5 (count conditions)))
            (doseq [condition conditions]
              (is (contains? #{:kernel-body-index-precondition :kernel-body-intrinsic-overflow}
                             (try (kernel-body/validate!
                                    (assoc-in candidate [:attributes :required-preconditions]
                                              (vec (remove #{condition} conditions)))) :accepted
                                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
          (is (= :kernel-body-intrinsic-overflow result)
              (str "unknown lower and inclusive loops do not acquire the exclusive entered proof: " result)))))))

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
        positive (remaining-product-artifact base types)
        bindings (fn [n ss seq-len other]
                   (fn [id] (case id _n_bound n ss ss seq-len seq-len other other slab 1 head-dim 1)))]
    (is (= positive (kart/validate! positive)))
    (doseq [[position r2] (map-indexed vector ['(rem r1 other) '(rem q1 seq-len)])]
      (let [different (-> base (update :scalars conj 'other)
                          (assoc-in [:scalar-region :locals 3 :init] r2))
            artifact (remaining-product-artifact different types)
            computes (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                   (contains? #{:quot :rem} (get-in % [:expression :op])))
                             (tree-seq coll? seq (get-in artifact [:attributes :kernel-body])))
            [q1 r1 q2 actual-r2] computes
            actual-arguments (get-in actual-r2 [:expression :arguments])]
        (is (= 4 (count computes)))
        (is (= (get-in (if (zero? position) r1 q1) [:result :id]) (first actual-arguments)))
        (is (= (if (zero? position) 'other 'seq-len) (second actual-arguments)))
        (is (not= (get-in q2 [:expression :arguments]) actual-arguments)
            "the emitted source keeps the mismatched operands rather than replacing the remainder")
        (assert-independent-address-contract artifact)
        (is (precondition/check! (:preconditions artifact) (bindings 1 1 1 1)))
        (is (precondition/check! (:preconditions artifact)
                                (bindings 0 Long/MIN_VALUE Long/MIN_VALUE Long/MIN_VALUE)))
        (doseq [divisor [0 -1]]
          (is (thrown? clojure.lang.ExceptionInfo
                       (precondition/check! (:preconditions artifact)
                         (if (zero? position) (bindings 1 1 1 divisor)
                             (bindings 1 1 divisor 1))))))
        (is (thrown? ArithmeticException
                     (precondition/check! (:preconditions artifact)
                       (if (zero? position) (bindings Long/MAX_VALUE 3 1 3)
                           (bindings Long/MAX_VALUE 1 3 1))))
            "this binding violates the actual retained conservative sum cap, not a claimed device counterexample")))))

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


(defn- ordinary-typed-quotient-address-operation [coordinate]
  (assoc (remaining-product-map nil #{'numerator 'divisor})
         :inputs #{'fine}
         :scalar-region
         {:locals [{:id 'q :dtype :long :init '(quot numerator divisor)}
                   {:id 'alias :dtype :long :init '(long q)}]
          :result (list 'aget 'fine coordinate)}))

(deftest typed-quotient-result-retains-its-declared-range-at-an-exact-identity-cast
  (let [captured (atom nil)
        original kernel-body/validate!
        result (with-redefs [kernel-body/validate!
                            (fn [checked]
                              (reset! captured checked)
                              (original checked))]
                 (try
                   {:scheduled (segmap-body/schedule
                                (ordinary-typed-quotient-address-operation 'alias)
                                {:scalar-types {'n :long 'numerator :long 'divisor :long}
                                 :array-types {'fine :double 'output :double}})}
                   (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))
        quotient (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                      (= :quot (get-in % [:expression :op])))
                                (tree-seq coll? seq @captured)))]
    (is (some? quotient) "the actual typed quotient reaches independent body validation")
    (is (= :long (get-in quotient [:expression :result-type])))
    (is (= {} (get-in quotient [:expression :options]))
        "optional range bookkeeping does not certify or change division semantics")
    (is (nil? (:error result)) (str "same-Long identity cast retains declared range: " (:error result)))
    (when-let [scheduled (:scheduled result)]
      (is (= (:body scheduled) (original (:body scheduled)))))))

(deftest typed-quotient-range-fallback-does-not-certify-overflowing-address-arithmetic
  (let [operation (ordinary-typed-quotient-address-operation '(+ alias (long 1)))
        options {:scalar-types {'n :long 'numerator :long 'divisor :long}
                 :array-types {'fine :double 'output :double}}
        scheduled (segmap-body/schedule operation options)
        operations (tree-seq coll? seq (:body scheduled))
        load (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                  (= 'fine (:buffer %))) operations))
        coordinate (first (:coordinates load))
        compute (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                     (= coordinate (get-in % [:result :id]))) operations))]
    (is (symbol? coordinate))
    (is (= :+ (get-in compute [:expression :op])))
    (is (= :long (get-in compute [:expression :result-type])))
    (is (= :trap (get-in compute [:expression :options :overflow]))
        "unknown quotient range does not certify unchecked addition; actual source checks remain")
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (doseq [target [:opencl-intel :cuda :hip]]
      (is (str/includes? (:source (sg/generate-scheduled-segmap-kernel operation
                                   :scalar-types (:scalar-types options)
                                   :array-types (:array-types options) :target-dialect target))
                         "rstr_trap_add_i64")))
    (is (= :kernel-body-c-trap-unsupported
           (try (sg/generate-scheduled-segmap-kernel operation
                  :scalar-types (:scalar-types options) :array-types (:array-types options)
                  :target-dialect :opencl-portable)
                nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
  (doseq [divisor [{:lower 0 :upper 0} {:lower -1 :upper -1}
                   (scalar-range/for-dtype :long)]]
    (is (nil? (scalar-range/quotient [(scalar-range/for-dtype :long) divisor]))
        "unknown or exceptional divisor domains acquire no interval proof")))


(defn- nested-quotient-numerator-address-operation []
  (let [typed-long (fn [form]
                     (walk/postwalk #(if (seq? %) (with-meta % {:raster.type/tag 'long}) %) form))]
    (assoc (remaining-product-map nil #{'slab 'group 'other 'different})
           :inputs #{'fine}
           :scalar-region
           {:locals [{:id 'q :dtype :long :init (typed-long '(quot (long row) slab))}
                     {:id 'r :dtype :long :init (typed-long '(rem (long row) slab))}
                     {:id 'g :dtype :long :init (typed-long '(quot q group))}]
            :result '(aget fine (+ (* g slab) r))})))

(deftest nested-quotient-numerator-retains-its-actual-identity-and-activation
  (let [operation (nested-quotient-numerator-address-operation)
        result (try
                 {:scheduled (segmap-body/schedule
                              operation
                              {:scalar-types {'n :long 'slab :long 'group :long 'other :long 'different :long}
                               :array-types {'fine :double 'output :double}})}
                 (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
    (is (nil? (:error result))
        (str "actual broadcast q=quot(row,slab),g=quot(q,group),r=rem(row,slab) address must close: " (:error result)))
    (when-let [error (:error result)]
      (is (= :kernel-body-index-precondition (:reason error))
          "the retained complete chain reaches independent address closure, not a setup decline"))
    (when-let [scheduled (:scheduled result)]
      (let [checked (:body scheduled)
            computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                             (tree-seq coll? seq checked))
            quotients (filter #(= :quot (get-in % [:expression :op])) computes)
            inner (last quotients)
            outer (first quotients)
            options (get-in inner [:expression :options])
            conditions (get-in checked [:attributes :required-preconditions])]
        (is (= 2 (count quotients)))
        (is (= (get-in outer [:result :id]) (first (get-in inner [:expression :arguments]))))
        (is (= {} options))
        (is (= checked (kernel-body/validate! checked)))
        (is (precondition/check! conditions
                                (fn [id] (case id _n_bound 0 slab Long/MIN_VALUE group Long/MIN_VALUE
                                               other Long/MIN_VALUE different Long/MIN_VALUE))))
        (doseq [replacement ['different 'other]]
          (let [position (if (= replacement 'different) 0 1)
                changed (walk/postwalk
                         (fn [value]
                           (if (= value inner)
                             (update-in value [:expression :arguments] assoc position replacement)
                             value)) checked)
                reason (try (kernel-body/validate! changed) nil
                            (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))]
            (is (contains? #{:kernel-body-index-precondition :kernel-body-required-preconditions
                             :kernel-body-scalar-precondition} reason)
                (str "changed actual numerator/divisor cannot inherit original obligations: " reason))))))))


(deftest different-origin-nonnegative-address-sum-requires-a-complete-proof
  (let [captured (atom nil)
        original kernel-body/validate!
        _ (with-redefs [kernel-body/validate! (fn [checked]
                                               (reset! captured checked)
                                               (original checked))]
            (try
              (segmap-body/schedule
               (nested-quotient-numerator-address-operation)
               {:scalar-types {'n :long 'slab :long 'group :long 'other :long 'different :long}
                :array-types {'fine :double 'output :double}})
              (catch clojure.lang.ExceptionInfo _ nil)))
        outer-upper (klaunch/maximum 0 (klaunch/sum
                                       (klaunch/ceil-div '_n_bound (klaunch/maximum 'slab 1)) -1))
        inner-upper (klaunch/maximum 0 (klaunch/sum
                                       (klaunch/ceil-div (klaunch/sum outer-upper 1)
                                                        (klaunch/maximum 'group 1)) -1))
        ;; Independently declare the current child multiplication obligations. The
        ;; validator still reconstructs them; this does not certify the parent sum.
        required (vec (distinct (concat (get-in @captured [:attributes :required-preconditions])
                                        [{:expression 'slab :op :>= :value 0}
                                         {:expression (klaunch/product inner-upper
                                                        (klaunch/product (klaunch/minimum '_n_bound 1) 'slab))
                                          :op :<= :value Long/MAX_VALUE}])))
        checked (assoc-in @captured [:attributes :required-preconditions] required)
        load (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                  (= 'fine (:buffer %))) (tree-seq coll? seq checked)))
        coordinate (first (:coordinates load))
        child (first (:arguments coordinate))
        child-only (walk/postwalk #(if (= % load) (assoc % :coordinates [child]) %) checked)
        result (try {:body (original checked)}
                    (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
    (is (= :add (:op coordinate)))
    (is (= :mul (:op child)))
    (is (= child-only (original child-only))
        "the actual child multiply is independently valid before claiming anything about the sum")
    (is (nil? (:error result))
        (str "different checked origins with the same activation require an independent combined bound: " (:error result)))
    (when-let [error (:error result)]
      (is (= :kernel-body-index-precondition (:reason error)))
      (is (= :add (get-in error [:expression :op]))
          "the discrimination reaches the parent addition after valid child obligations"))))


(deftest compositional-address-obligations-cannot-be-dropped-or-forged
  (let [scheduled (segmap-body/schedule
                   (nested-quotient-numerator-address-operation)
                   {:scalar-types {'n :long 'slab :long 'group :long 'other :long 'different :long}
                    :array-types {'fine :double 'output :double}})
        checked (:body scheduled)
        conditions (get-in checked [:attributes :required-preconditions])
        caps (filter #(and (= :<= (:op %)) (= Long/MAX_VALUE (:value %))) conditions)
        load (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                  (= 'fine (:buffer %))) (tree-seq coll? seq checked)))
        coordinate (first (:coordinates load))
        child (first (:arguments coordinate))
        rejection (fn [body]
                    (try (kernel-body/validate! body) nil
                         (catch clojure.lang.ExceptionInfo error (ex-data error))))]
    (is (= checked (kernel-body/validate! checked)))
    (is (= 2 (count caps)) "separate child multiplication and final addition bounds are retained")
    (doseq [condition conditions]
      (let [changed (assoc-in checked [:attributes :required-preconditions]
                              (vec (remove #{condition} conditions)))]
        (is (= :kernel-body-index-precondition (:reason (rejection changed)))
            (str "each inherited/child/parent obligation is independently required: " condition))))
    (let [sum-cap (first (filter #(instance? raster.compiler.ir.kernel_launch.Sum (:expression %)) caps))
          without-sum (assoc-in checked [:attributes :required-preconditions]
                                (vec (remove #{sum-cap} conditions)))
          child-only (walk/postwalk #(if (= % load) (assoc % :coordinates [child]) %) without-sum)]
      (is (some? sum-cap))
      (is (= child-only (kernel-body/validate! child-only)))
      (is (= :add (get-in (rejection without-sum) [:expression :op]))))
    (let [without-guard (update checked :operations
                                #(vec (mapcat (fn [operation]
                                                (if (instance? raster.compiler.ir.kernel_body.Guard operation)
                                                  (:operations operation) [operation])) %)))
          forged (with-meta (assoc-in without-guard [:attributes :active-upper] '_n_bound)
                   {:active-upper '_n_bound :quotient-address-origin? false})]
      (is (zero? (count (filter #(instance? raster.compiler.ir.kernel_body.Guard %)
                               (tree-seq coll? seq without-guard)))))
      (is (= :kernel-body-index-precondition (:reason (rejection without-guard))))
      (is (= :kernel-body-index-precondition (:reason (rejection forged)))
          "caller metadata cannot supply the removed actual activation"))
    (let [different-activation
          (update checked :masks
                  (fn [masks]
                    (walk/postwalk
                     (fn [value]
                       (if (and (instance? raster.compiler.ir.kernel_body.IndexCast value)
                                (= '_n_bound (:argument value)))
                         (assoc value :argument 'other) value)) masks)))]
      (is (= :kernel-body-index-precondition (:reason (rejection different-activation)))
          "a different actual guard bound cannot borrow the original domain obligations"))))

(deftest independently-nonnegative-quotient-address-sums-still-reject-active-overflow
  (let [typed-long (fn [form]
                     (walk/postwalk #(if (seq? %) (with-meta % {:raster.type/tag 'long}) %) form))
        operation (assoc (remaining-product-map nil #{'d1 'd2}) :inputs #{'fine}
                         :scalar-region
                         {:locals [{:id 'q1 :dtype :long :init (typed-long '(quot (long row) d1))}
                                   {:id 'q2 :dtype :long :init (typed-long '(quot (long row) d2))}]
                          :result '(aget fine (+ q1 q2))})
        scheduled (segmap-body/schedule operation
                                       {:scalar-types {'n :long 'd1 :long 'd2 :long}
                                        :array-types {'fine :double 'output :double}})
        conditions (get-in scheduled [:body :attributes :required-preconditions])
        bindings (fn [n d1 d2] (fn [id] (case id _n_bound n d1 d1 d2 d2)))]
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (precondition/check! conditions (bindings 0 Long/MIN_VALUE Long/MIN_VALUE)))
    (is (precondition/check! conditions (bindings 1 Long/MAX_VALUE Long/MAX_VALUE)))
    (is (> (*' 2 (dec Long/MAX_VALUE)) Long/MAX_VALUE)
        "the actual active row MAX-1 produces two individually fitting values whose sum overflows")
    (is (thrown? ArithmeticException
                 (precondition/check! conditions (bindings Long/MAX_VALUE 1 1))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (precondition/check! conditions (bindings 1 0 1))))))


(deftest computed-uniform-quotient-loop-keeps-its-signed-evaluated-prefix
  (let [typed-long (fn [form]
                     (walk/postwalk #(if (seq? %) (with-meta % {:raster.type/tag 'long}) %) form))
        operation (assoc (remaining-product-map nil #{'in 'out})
                         :space (segop/make-seg-space 'ro 'n) :inputs #{'fine}
                         :scalar-region
                         {:locals [{:id 'o :dtype :long :init (typed-long '(rem (long ro) out))}
                                   {:id 'nsb :dtype :long :init (typed-long '(quot in (long 256)))}
                                   {:id 'sbb :dtype :long :init (typed-long '(* o nsb))}]
                          :result '(fold {:accumulator acc :index sb :identity 0.0 :lower 0
                                          :extent nsb :dtype :double :association :ordered}
                                         (lambda [acc sb]
                                           (region [] [(+ acc (aget fine (+ sbb sb)))])))})
        captured (atom nil)
        original kernel-body/validate!
        result (with-redefs [kernel-body/validate! (fn [checked]
                                                   (reset! captured checked)
                                                   (original checked))]
                 (try {:scheduled (segmap-body/schedule operation
                                         {:scalar-types {'n :long 'in :long 'out :long}
                                          :array-types {'fine :double 'output :double}})}
                      (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))
        computes (filter #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                         (tree-seq coll? seq @captured))
        quotient (first (filter #(= :quot (get-in % [:expression :op])) computes))
        prefix (first (filter #(= :* (get-in % [:expression :op])) computes))
        loop (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                           (tree-seq coll? seq @captured)))]
    (is (= {} (get-in quotient [:expression :options])))
    (is (= :long (get-in prefix [:expression :result-type])))
    (is (= :trap (get-in prefix [:expression :options :overflow]))
        "the actual signed prefix is checked before entering the computed-bound loop")
    (is (some? loop) "the ordered [0,quot(in,256)) loop remains explicit")
    (is (nil? (:error result))
        (str "the actual Q4_K computed factor/prefix/loop/load chain requires a complete proof: " (:error result)))
    (when-let [error (:error result)]
      (is (= :kernel-body-index-precondition (:reason error)))
      (is (= :add (get-in error [:expression :op]))))))


(defn- checked-loaded-offset-operation []
  (let [typed-long (fn [form]
                     (walk/postwalk #(if (seq? %) (with-meta % {:raster.type/tag 'long}) %) form))]
    (assoc (remaining-product-map nil #{'n-slices 'total-dim 'slice-dim})
           :inputs #{'indices 'fine}
           :scalar-region
           {:locals [{:id 'e :dtype :long :init (typed-long '(quot (long row) n-slices))}
                     {:id 's :dtype :long :init (typed-long '(rem (long row) n-slices))}
                     {:id 'ia :dtype :long :init '(aget indices e)}
                     {:id 'off-a :dtype :long :init (typed-long '(+ (* ia (int total-dim))
                                                                  (* s (int slice-dim))))}]
            :result '(fold {:accumulator acc :index d :identity 0.0 :lower 0
                            :extent slice-dim :dtype :double :association :ordered}
                           (lambda [acc d]
                             (region [] [(+ acc (aget fine (+ off-a d)))])))})))

(deftest checked-loaded-long-addresses-retain-scalar-ssa-and-target-traps
  (let [operation (checked-loaded-offset-operation)
        types {'n :long 'n-slices :long 'total-dim :long 'slice-dim :long}
        arrays {'indices :long 'fine :double 'output :double}
        captured (atom nil)
        original kernel-body/validate!
        result (with-redefs [kernel-body/validate! (fn [checked]
                                                   (reset! captured checked)
                                                   (original checked))]
                 (try {:scheduled (segmap-body/schedule operation
                                         {:scalar-types types :array-types arrays})}
                      (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))
        load (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                  (= 'fine (:buffer %))) (tree-seq coll? seq @captured)))
        coordinate (first (:coordinates load))
        address-compute (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                             (= coordinate (get-in % [:result :id])))
                                       (tree-seq coll? seq @captured)))]
    (is (some? load) "the actual Long-index loaded prefix and entered-loop load reach body validation")
    (is (symbol? coordinate) "the source checked address is a typed scalar SSA result, not raw C index addition")
    (is (nil? (:error result)) (str "signed checked source address must remain typed: " (:error result)))
    (when-let [scheduled (:scheduled result)]
      (is (= (:body scheduled) (original (:body scheduled))))
      (is (= :+ (get-in address-compute [:expression :op])))
      (is (= :long (get-in address-compute [:expression :result-type])))
      (is (= :trap (get-in address-compute [:expression :options :overflow])))
      (doseq [target [:opencl-intel :cuda :hip]]
        (let [artifact (sg/generate-scheduled-segmap-kernel operation
                        :scalar-types types :array-types arrays :target-dialect target)]
          (is (str/includes? (:source artifact) "rstr_trap_add_i64"))))
      (is (= :kernel-body-c-trap-unsupported
             (try (sg/generate-scheduled-segmap-kernel operation
                    :scalar-types types :array-types arrays :target-dialect :opencl-portable)
                  nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest checked-address-selection-preserves-reduction-owner-coordinate-mapping
  (let [source '(aget fine (+ original-index offset))
        calls (atom [])
        mapped (kernel-body/expression :add 'element-index 'offset)
        lowered (segred-body/lower-element-operations
                 source
                 {:index 'original-index :coordinate 'element-index :dtype :double
                  :arrays #{'fine} :array-types {'fine :double}
                  :scalars #{'offset} :scalar-types {'offset :long}
                  :declared-result-dtype :double
                  :coordinate-lower (fn [coordinate]
                                      (swap! calls conj coordinate)
                                      (when (= '(+ original-index offset) coordinate) mapped))})
        load (first (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                            (:operations lowered)))]
    (is (= ['(+ original-index offset)] @calls))
    (is (= [mapped] (:coordinates load))
        "the actual reduction owner retains its admitted element-index mapping")
    (is (not-any? #{'original-index} (tree-seq coll? seq (:operations lowered))))
    (is (= :indexed-load
           (try (segred-body/lower-element-operations
                 source
                 {:index 'original-index :coordinate 'element-index :dtype :double
                  :arrays #{'fine} :array-types {'fine :double}
                  :scalars #{'offset} :scalar-types {'offset :long}
                  :declared-result-dtype :double :coordinate-lower (constantly nil)})
                nil
                (catch clojure.lang.ExceptionInfo error (:missing-rule (ex-data error)))))
        "source materialization cannot bypass the owner's coordinate admission")))

(deftest checked-address-selection-keeps-unbound-staged-geometry-structural
  (let [coordinate (kernel-body/expression :add 'staged-axis (kernel-body/literal 1 :long))
        seen (atom [])
        lowerer (scalar-body/make-lowerer
                 {:arrays #{'fine} :array-types {'fine :double} :scalar-types {}
                  :index-scope #{'staged-axis}
                  :lower-index (fn [source _]
                                 (swap! seen conj source)
                                 coordinate)
                  :decline! (fn [rule message data]
                              (throw (ex-info message (assoc data :missing-rule rule))))})
        source '(aget fine (+ staged-axis (long 1)))
        lowered ((:lower lowerer) source :double {})
        load (first (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                            (:operations lowered)))]
    (is (= ['(+ staged-axis (long 1))] @seen))
    (is (= [coordinate] (:coordinates load)))
    (is (not-any? #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                  (:operations lowered))
        "unbound scheduling geometry is not guessed to be a typed source scalar")))

(defn- checked-store-address-operation [kind]
  (let [typed-long #(walk/postwalk (fn [node] (if (seq? node)
                                             (with-meta node {:raster.type/tag 'long}) node)) %)
        scatter? (= :scatter kind)
        locals (if scatter?
                 [{:id 'block :dtype :long :init (typed-long '(quot (long row) width))}
                  {:id 'r :dtype :long :init (typed-long '(rem (long row) width))}
                  {:id 'destination :dtype :int :init '(aget indices block)}]
                 [{:id 'half-width :dtype :long :init (typed-long '(quot head-dim (long 2)))}
                  {:id 'per-row :dtype :long :init (typed-long '(* heads half-width))}
                  {:id 't :dtype :long :init (typed-long '(quot (long row) per-row))}
                  {:id 'rest0 :dtype :long :init (typed-long '(rem (long row) per-row))}
                  {:id 'h :dtype :long :init (typed-long '(quot rest0 half-width))}
                  {:id 'i :dtype :long :init (typed-long '(rem rest0 half-width))}
                  {:id 'base :dtype :long :init (typed-long '(+ (* t (* heads head-dim))
                                                             (* h head-dim)))}])
        result (if scatter?
                 '(if (>= destination 0)
                    (if (< destination indexed-blocks)
                      (clojure.core/aset output (+ (* destination width) r) (aget fine row))))
                 '(do (clojure.core/aset output (+ base i) (aget fine (+ base i)))
                      (clojure.core/aset output (+ (+ base i) half-width)
                                        (aget fine (+ (+ base i) half-width)))))]
    (assoc (remaining-product-map result (if scatter? #{'width 'indexed-blocks} #{'heads 'head-dim}))
           :out-sym nil :inputs (if scatter? #{'indices 'fine} #{'fine})
           :scalar-region {:locals locals :result result})))

(deftest checked-scatter-store-retains-its-loaded-index-and-branch-placement
  (let [captured (atom nil) original kernel-body/validate!
        result (with-redefs [kernel-body/validate! (fn [body] (reset! captured body) (original body))]
                 (try {:scheduled (segmap-body/schedule (checked-store-address-operation :scatter)
                                    {:scalar-types {'n :long 'width :long 'indexed-blocks :long}
                                     :array-types {'indices :int 'fine :double 'output :double}})}
                      (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))
        nodes (tree-seq coll? seq @captured)]
    (is (= 1 (count (filter #(instance? raster.compiler.ir.kernel_body.ScalarStore %) nodes))))
    (is (= 2 (count (filter #(instance? raster.compiler.ir.kernel_body.IfRegion %) nodes))))
    (is (some #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                    (= 'indices (:buffer %)) (= :int (get-in % [:result :type]))) nodes))
    (is (nil? (:error result)) (str "actual scatter destination arithmetic remains typed: " (:error result)))
    (when-let [scheduled (:scheduled result)]
      (let [inner (first (filter #(and (instance? raster.compiler.ir.kernel_body.IfRegion %)
                                      (some (fn [node] (instance? raster.compiler.ir.kernel_body.ScalarStore node))
                                            (:then-operations %))) nodes))
            operations (:then-operations inner)
            store (first (filter #(instance? raster.compiler.ir.kernel_body.ScalarStore %) operations))
            coordinate (first (:coordinates store))
            compute (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                         (= coordinate (get-in % [:result :id]))) operations))
            payload (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                         (= (:value store) (get-in % [:result :id]))) operations))]
        (is (symbol? coordinate))
        (is (= :trap (get-in compute [:expression :options :overflow])))
        (let [definitions (into {} (for [node nodes
                                         :when (instance? raster.compiler.ir.kernel_body.ScalarCompute node)]
                                     [(get-in node [:result :id]) node]))
              address-computes
              (letfn [(dependencies [id]
                        (when-let [node (get definitions id)]
                          ;; Source q/r already execute before the destination branch.
                          ;; Newly materialized address arithmetic and casts do not.
                          (when-not (contains? #{:quot :rem} (get-in node [:expression :op]))
                            (cons node (mapcat dependencies (get-in node [:expression :arguments]))))))]
                (set (dependencies coordinate)))
              multiply (first (filter #(= :* (get-in % [:expression :op])) address-computes))
              outer (first (filter #(and (instance? raster.compiler.ir.kernel_body.IfRegion %)
                                         (some #{inner} (:then-operations %))) nodes))]
          (is (some? multiply) "the loaded destination multiplication is an actual dependency")
          (is (every? (set operations) address-computes)
              "every generated address arithmetic/cast dependency stays in the inner branch")
          (is (some? outer) "the inner destination-limit branch remains under the nonnegative guard")
          (is (some #{'width} (mapcat #(get-in % [:expression :arguments]) address-computes)))
          (is (every? #(< (.indexOf operations %) (.indexOf operations payload)) address-computes)))
        (is (< (.indexOf operations compute) (.indexOf operations payload) (.indexOf operations store))
            "destination arithmetic stays inside both source branches, before value evaluation/store")
        (is (= (:body scheduled) (original (:body scheduled))))))
    (when-let [error (:error result)]
      (is (= :kernel-body-index-precondition (:reason error))))))

(deftest checked-paired-stores-retain-both-source-addresses
  (let [captured (atom nil) original kernel-body/validate!
        result (with-redefs [kernel-body/validate! (fn [body] (reset! captured body) (original body))]
                 (try {:scheduled (segmap-body/schedule (checked-store-address-operation :paired)
                                    {:scalar-types {'n :long 'heads :long 'head-dim :long}
                                     :array-types {'fine :double 'output :double}})}
                      (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))
        nodes (tree-seq coll? seq @captured)]
    (is (= 2 (count (filter #(instance? raster.compiler.ir.kernel_body.ScalarStore %) nodes))))
    (is (= 2 (count (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                 (= 'fine (:buffer %))) nodes))))
    (is (nil? (:error result)) (str "both actual paired destination additions remain checked: " (:error result)))
    (when-let [scheduled (:scheduled result)]
      (doseq [store (filter #(instance? raster.compiler.ir.kernel_body.ScalarStore %) nodes)]
        (let [coordinate (first (:coordinates store))
              compute (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                           (= coordinate (get-in % [:result :id]))) nodes))]
          (is (symbol? coordinate))
          (is (= :trap (get-in compute [:expression :options :overflow])))
          (is (= :map-active (:predicate store)))))
      (is (= (:body scheduled) (original (:body scheduled)))))
    (when-let [error (:error result)]
      (is (= :kernel-body-index-precondition (:reason error))))))

(deftest devirtualized-checked-address-uses-semantic-operands-only
  (let [coordinate (with-meta (list '.invk 'implementation-receiver 'alias '(long 1))
                     {:raster.op/original 'raster.numeric/+ :raster.type/tag 'long})
        scheduled (segmap-body/schedule (ordinary-typed-quotient-address-operation coordinate)
                    {:scalar-types {'n :long 'numerator :long 'divisor :long}
                     :array-types {'fine :double 'output :double}})
        nodes (tree-seq coll? seq (:body scheduled))
        load (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                                  (= 'fine (:buffer %))) nodes))
        id (first (:coordinates load))
        compute (first (filter #(and (instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                                     (= id (get-in % [:result :id]))) nodes))]
    (is (symbol? id))
    (is (= :+ (get-in compute [:expression :op])))
    (is (= :trap (get-in compute [:expression :options :overflow])))
    (is (not-any? #{'implementation-receiver} nodes))
    (is (= (:body scheduled) (kernel-body/validate! (:body scheduled))))
    (is (= :unbound-index-symbol
           (try (segmap-body/schedule
                 (ordinary-typed-quotient-address-operation
                  (with-meta (list '.invk 'implementation-receiver 'alias 'unknown-operand)
                    {:raster.op/original 'raster.numeric/+ :raster.type/tag 'long}))
                 {:scalar-types {'n :long 'numerator :long 'divisor :long}
                  :array-types {'fine :double 'output :double}})
                nil
                (catch clojure.lang.ExceptionInfo error (:missing-rule (ex-data error)))))
        "excluding an implementation receiver does not authorize an unknown semantic operand")))

(deftest guarded-affine-address-keeps-product-and-loop-sum-authority
  (let [source '(let* [effect (raster.par/map-void!
                              i rows
                              (if (> enabled 0)
                                (dotimes [j width]
                                  (aset out (+ (* i width) j)
                                        (aget input (+ (* i width) j))))))] effect)
        types {'rows :long 'width :long 'enabled :int}
        result (typed-route/attempt source :float {'out :float 'input :float}
                                   {:scalar-types types})
        program (-> result :program :equations first :algorithm)
        operation (first (lower/lower-typed-effect-map program :ze:0 :dtype :float))
        artifact (sg/generate-scheduled-segmap-kernel
                  operation :dtype :float :target-dialect :opencl-portable
                  :array-types {'out :float 'input :float} :scalar-types types)
        checked (get-in artifact [:attributes :kernel-body])
        required (kernel-body/required-preconditions checked)
        affine (kernel-body/affine-product-preconditions '_n_bound 'width)
        arguments (fn [n width]
                    (mapv (fn [slot]
                            (case (:name slot)
                              _n_bound {:type :long :value n}
                              width {:type :long :value width}
                              enabled {:type :int :value 1}
                              nil)) (:abi artifact)))]
    (is (= checked (kernel-body/validate! checked)))
    (is (every? (set required) affine))
    (is (kcall/validate-preconditions! artifact (arguments 2 3)))
    (is (kcall/validate-preconditions! artifact (arguments 1 Long/MAX_VALUE)))
    (is (thrown? ArithmeticException
                 (kcall/validate-preconditions! artifact (arguments 2 Long/MAX_VALUE))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (kcall/validate-preconditions! artifact (arguments 0 -1)))
        "the conservative affine cohort keeps its nonnegative factor requirement")
    (let [prefix-source '(let* [effect (raster.par/map-void!
                                        i rows
                                        (let* [^long offset (* i width)]
                                          (dotimes [j width]
                                            (aset out (+ offset j) (aget input (+ offset j))))))] effect)
          prefix-result (typed-route/attempt prefix-source :float {'out :float 'input :float}
                                             {:scalar-types types})
          prefix-operation (first (lower/lower-typed-effect-map
                                   (-> prefix-result :program :equations first :algorithm)
                                   :ze:0 :dtype :float))
          prefix-artifact (sg/generate-scheduled-segmap-kernel
                           prefix-operation :dtype :float :target-dialect :opencl-portable
                           :array-types {'out :float 'input :float} :scalar-types types)
          prefix-arguments (mapv (fn [slot]
                                   (case (:name slot)
                                     _n_bound {:type :long :value 1}
                                     width {:type :long :value -1}
                                     enabled {:type :int :value 1} nil)) (:abi prefix-artifact))
          prefix-nodes (kernel-body-operations prefix-artifact)]
      (is (some #(and (= "ScalarCompute" (operation-kind %))
                      (= :* (get-in % [:expression :op]))) prefix-nodes))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kcall/validate-preconditions! prefix-artifact prefix-arguments))
          "the actual product evaluated before an empty loop retains its factor contract"))
    (doseq [condition affine]
      (let [missing (update-in checked [:attributes :required-preconditions]
                               #(vec (remove #{condition} %)))]
        (is (thrown? clojure.lang.ExceptionInfo (kernel-body/validate! missing))
            "independent validation reconstructs each actual product obligation")))))

(deftest affine-product-transfer-retains-the-exact-matched-orientation
  (let [full {:lower Long/MIN_VALUE :upper Long/MAX_VALUE}
        values {'factor {:type :long :range full :abi-scalar-identity 'factor :active-upper 'wrong-n}
                'row {:type :long :range {:lower 0 :upper Long/MAX_VALUE} :active-upper 'n}}
        transfer (kernel-body/conditional-integer-transfer
                  (kernel-body/scalar-expression :* :long ['factor 'row] {}) values)]
    (is (= 'row (get-in transfer [:fact :origin])))
    (is (= 'n (get-in transfer [:fact :bound])))
    (is (= 'factor (get-in transfer [:fact :factor])))
    (is (= (kernel-body/affine-product-preconditions 'n 'factor)
           (:required-preconditions transfer)))
    (is (nil? (kernel-body/conditional-integer-transfer
               (kernel-body/scalar-expression :* :long ['factor 'row] {})
               (update values 'row dissoc :active-upper))))))

(defn- guarded-affine-effect-body []
  (let [source '(let* [effect (raster.par/map-void!
                              i rows (if (> enabled 0)
                                       (dotimes [j width]
                                         (aset out (+ (* i width) j)
                                               (aget input (+ (* i width) j))))))] effect)
        types {'rows :long 'width :long 'enabled :int}
        result (typed-route/attempt source :float {'out :float 'input :float} {:scalar-types types})
        operation (first (lower/lower-typed-effect-map
                          (-> result :program :equations first :algorithm) :ze:0 :dtype :float))]
    (get-in (sg/generate-scheduled-segmap-kernel operation :dtype :float :target-dialect :opencl-portable
              :array-types {'out :float 'input :float} :scalar-types types) [:attributes :kernel-body])))

(defn- affine-pointer-node? [node]
  (or (instance? raster.compiler.ir.kernel_body.ScalarLoad node)
      (instance? raster.compiler.ir.kernel_body.ScalarStore node)))

(defn- affine-address-alias-body [checked cast?]
  (let [serial (atom 0)]
    (walk/postwalk
     (fn [node]
       (if (instance? raster.compiler.ir.kernel_body.ForLoop node)
         (update node :operations
           #(vec (mapcat (fn [operation]
                           (if (affine-pointer-node? operation)
                             (let [id (symbol (str "permanent-address-alias-" (swap! serial inc)))
                                   expression (first (:coordinates operation))]
                               [(kernel-body/->IndexCompute id expression)
                                (assoc operation :coordinates
                                       [(if cast? (kernel-body/index-cast id :long :exact) id)])])
                             [operation])) %))) node)) checked)))

(deftest raw-affine-address-proof-cannot-disappear-at-an-index-alias
  (let [checked (guarded-affine-effect-body)
        unsafe (assoc-in checked [:attributes :required-preconditions] [])
        zero (kernel-body/index-cast 0 :long :exact)
        hidden (kernel-body/expression :min zero
                 (kernel-body/expression :max zero (kernel-body/expression :mul 'width 'width)))
        top (-> (walk/postwalk #(if (affine-pointer-node? %)
                                 (assoc % :coordinates ['permanent-top-alias]) %) unsafe)
                (update :indices conj (kernel-body/->IndexCompute 'permanent-top-alias hidden)))
        without-guard (update checked :operations
                        #(vec (mapcat (fn [operation]
                                       (if (instance? raster.compiler.ir.kernel_body.Guard operation)
                                         (:operations operation) [operation])) %)))
        different-bound (update checked :masks
                          (fn [masks]
                            (mapv (fn [mask]
                                    (update mask :predicates
                                      #(mapv (fn [predicate]
                                               (let [upper (second (:arguments predicate))]
                                                 (if (and (= :lt (:op predicate))
                                                          (or (= '_n_bound upper)
                                                              (= '_n_bound (:argument upper))))
                                                   (assoc-in predicate [:arguments 1]
                                                     (if (map? upper) (assoc upper :argument 'width) 'width))
                                                   predicate))) %))) masks)))]
    (is (= checked (kernel-body/validate! checked)))
    (doseq [cast? [false true]]
      (let [safe (affine-address-alias-body checked cast?)]
        (is (= safe (kernel-body/validate! safe)) "actual complete lexical obligations permit the alias")))
    (doseq [candidate [unsafe (affine-address-alias-body unsafe false)
                       (affine-address-alias-body unsafe true) top without-guard different-bound]]
      (is (= :kernel-body-index-precondition
             (try (kernel-body/validate! candidate) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "the same unchecked source tree cannot become a checked machine value through an alias"))
    (let [ordinary (-> (walk/postwalk #(if (affine-pointer-node? %)
                                       (assoc % :coordinates ['permanent-ordinary-index]) %) checked)
                       (update :indices conj
                         (kernel-body/->IndexCompute 'permanent-ordinary-index
                           (kernel-body/expression :mul (kernel-body/expression :add 2 3) 4))))]
      (is (= ordinary (kernel-body/validate! ordinary)) "independently bounded ordinary index arithmetic remains valid"))))

(deftest checked-scalar-casts-cannot-launder-an-earlier-unchecked-index
  (let [checked (guarded-affine-effect-body)
        unsafe (affine-address-alias-body (assoc-in checked [:attributes :required-preconditions] []) false)
        serial (atom 0)
        converted (walk/postwalk
          (fn [node]
            (if (instance? raster.compiler.ir.kernel_body.ForLoop node)
              (update node :operations
                #(vec (mapcat (fn [operation]
                                (if (affine-pointer-node? operation)
                                  (let [old (first (:coordinates operation))
                                        fp (symbol (str "address-floating-" (swap! serial inc)))
                                        integer (symbol (str "address-integer-" (swap! serial inc)))]
                                    [(kernel-body/->ScalarCompute (kernel-body/value fp :double)
                                      (kernel-body/scalar-expression :cast :double [old]
                                        {:rounding :nearest-even :overflow :exact}))
                                     (kernel-body/->ScalarCompute (kernel-body/value integer :long)
                                      (kernel-body/scalar-expression :cast :long [fp]
                                        {:rounding :toward-zero :overflow :exact}))
                                     (assoc operation :coordinates [integer])]) [operation])) %))) node)) unsafe)]
    (is (= :kernel-body-index-precondition
           (try (kernel-body/validate! converted) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "actual typed conversion contracts preserve unresolved source ancestry across floating values")))

(deftest loop-backedges-preserve-unchecked-index-history
  (let [checked (guarded-affine-effect-body)
        zero (kernel-body/index-cast 0 :long :exact)
        hidden (kernel-body/expression :min zero
                 (kernel-body/expression :max zero (kernel-body/expression :mul 'width 'width)))
        carry-body
        (fn [unsafe?]
          (walk/postwalk
           (fn [node]
             (if (instance? raster.compiler.ir.kernel_body.ForLoop node)
               (assoc node :iter-args [(kernel-body/->LoopArg (kernel-body/value 'address-carry :long)
                                                      (kernel-body/literal 0 :long))]
                          :results [(kernel-body/value 'address-loop-result :long)]
                          :operations
                          (vec (concat
                                 (map #(if (affine-pointer-node? %)
                                         (assoc % :coordinates ['address-carry]) %)
                                      (pop (:operations node)))
                                 [(kernel-body/->IndexCompute 'address-next (if unsafe? hidden zero))
                                  (kernel-body/->Yield ['address-next])]))) node)) checked))
        unsafe (carry-body true) safe (carry-body false)]
    (is (= safe (kernel-body/validate! safe)) "ordinary carry values keep the same loop and claim semantics")
    (is (= :kernel-body-index-precondition
           (try (kernel-body/validate! unsafe) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "the second iteration's address cannot ignore the actual raw backedge definition")))

(deftest zero-trip-loop-results-retain-the-actual-initial-index-history
  (let [checked (guarded-affine-effect-body)
        zero (kernel-body/index-cast 0 :long :exact)
        hidden (kernel-body/expression :min zero
                 (kernel-body/expression :max zero (kernel-body/expression :mul 'width 'width)))
        original-loop (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                                     (tree-seq coll? seq checked)))
        candidate (fn [expression]
                    (-> checked
                        (update :indices conj (kernel-body/->IndexCompute 'zero-trip-initial expression))
                        (update :operations into
                          [(assoc original-loop :index (kernel-body/value 'zero-trip-index :long)
                             :lower zero :upper zero :step 1
                             :iter-args [(kernel-body/->LoopArg (kernel-body/value 'zero-trip-carry :long)
                                                               'zero-trip-initial)]
                             :results [(kernel-body/value 'zero-trip-result :long)]
                             :operations [(kernel-body/->Yield [(kernel-body/literal 0 :long)])])
                           (kernel-body/->ScalarStore 'out ['zero-trip-result]
                                                       (kernel-body/literal 0 :float) :map-active)])))
        safe (candidate zero) unsafe (candidate hidden)]
    (is (= safe (kernel-body/validate! safe)))
    (is (= :kernel-body-index-precondition
           (try (kernel-body/validate! unsafe) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "zero iterations return the actual initial value, even when the unentered yield is ordinary")))

(deftest logical-long-launch-geometry-widens-before-every-arithmetic-prefix
  (let [scheduled (segmap-body/schedule
                    (ordinary-typed-quotient-address-operation '(long 0))
                    {:scalar-types {'n :long 'numerator :long 'divisor :long}
                     :array-types {'fine :double 'output :double}})
        checked (:body scheduled)
        indices (:indices checked)
        hardware (filter #(instance? raster.compiler.ir.kernel_body.IndexBinding %) indices)
        group (:id (first (filter #(= :group (:source %)) hardware)))
        local (:id (first (filter #(= :local (:source %)) hardware)))
        coordinate (:expression (first (filter #(instance? raster.compiler.ir.kernel_body.IndexCompute %) indices)))
        evaluate (fn evaluate [node values]
                   (cond (symbol? node) (get values node)
                         (integer? node) (bigint node)
                         (instance? raster.compiler.ir.kernel_body.IndexCast node)
                         (evaluate (:argument node) values)
                         (instance? raster.compiler.ir.kernel_body.IndexExpr node)
                         (apply ({:add +' :mul *'} (:op node))
                                (map #(evaluate % values) (:arguments node)))))
        nodes (tree-seq coll? seq coordinate)]
    (is (= :add (:op coordinate)))
    (is (= 3 (count (filter #(instance? raster.compiler.ir.kernel_body.IndexCast %) nodes))))
    (is (every? #(= :long (:dtype %))
                (filter #(instance? raster.compiler.ir.kernel_body.IndexCast %) nodes)))
    (is (= 2147483648N (evaluate coordinate {group 8388608 local 0})))
    (is (= (klaunch/geometry {:workgroup-size [256] :group-count [8388609]})
           (kernel-body/validate-launch-index-ranges!
             checked (klaunch/geometry {:workgroup-size [256] :group-count [8388609]}))))
    (is (= :kernel-launch-index-range
           (try (kernel-body/validate-launch-index-ranges!
                  checked (klaunch/geometry {:workgroup-size [256] :group-count [2147483649]})) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= checked (kernel-body/validate! checked)))))

(deftest strided-loop-entry-ranges-come-from-the-actual-exclusive-exit-contract
  (let [checked (guarded-affine-effect-body)
        template (first (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                                (tree-seq coll? seq checked)))
        scalar-var (ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!)
        original (var-get scalar-var)
        run (fn [lower upper step mode proof]
              (let [seen (atom nil)
                    dynamic? (= step :dynamic)
                    induction 'verified-strided-index
                    loop (assoc template :index (kernel-body/value induction :long)
                           :lower 'verified-strided-lower
                           :upper 'verified-strided-upper
                           :step (if dynamic? (kernel-body/index-cast 'verified-positive-step :long :exact) step)
                           :iter-args [] :results []
                           :attributes {:upper-bound mode}
                           :operations
                           [(kernel-body/->ScalarCompute (kernel-body/value 'verified-strided-copy :long)
                              (kernel-body/scalar-expression :+ :long [induction (kernel-body/literal 0 :long)]
                                {:overflow :no-overflow :proof (assoc proof :kind :typed-scalar-range)}))
                            (kernel-body/->Yield [])])
                    candidate (cond-> (assoc checked :operations
                                               [(kernel-body/->ScalarCompute
                                                  (kernel-body/value 'verified-strided-lower :long)
                                                  (kernel-body/cast-expression (kernel-body/literal lower :long) :long :exact :exact))
                                                (kernel-body/->ScalarCompute
                                                  (kernel-body/value 'verified-strided-upper :long)
                                                  (kernel-body/cast-expression (kernel-body/literal upper :long) :long :exact :exact))
                                                loop])
                                dynamic? (update :indices conj
                                           (kernel-body/->IndexBinding 'verified-positive-step :group-count 0)))
                    outcome (with-redefs-fn
                              {scalar-var (fn [expression values]
                                            (when (contains? values induction)
                                              (reset! seen (:range (get values induction))))
                                            (original expression values))}
                              #(try {:valid (kernel-body/validate! candidate)}
                                    (catch clojure.lang.ExceptionInfo error {:error (ex-data error)})))]
                (assoc outcome :range @seen)))
        exclusive (run -5 10 3 :exclusive {:lower -5 :upper 9})
        inclusive (run -5 10 3 :inclusive {:lower -5 :upper 9})
        dynamic (run -5 10 :dynamic :exclusive {:lower -5 :upper 10})
        empty (run 10 5 3 :exclusive {:lower 5 :upper 10})
        edge (run (- Long/MAX_VALUE 2) Long/MAX_VALUE 4 :exclusive
                  {:lower (- Long/MAX_VALUE 2) :upper Long/MAX_VALUE})]
    (is (some? (:valid exclusive)))
    (is (= {:lower -5 :upper 7} (:range exclusive)))
    (is (= :kernel-body-intrinsic-overflow (get-in inclusive [:error :reason])))
    (is (= {:lower -5 :upper 10} (:range inclusive)))
    (is (some? (:valid dynamic)))
    (is (= {:lower -5 :upper 10} (:range dynamic)))
    (is (some? (:valid empty)))
    (is (= {:lower 5 :upper 10} (:range empty)))
    (is (some? (:valid edge)) (pr-str edge))
    (is (= {:lower (- Long/MAX_VALUE 2) :upper (- Long/MAX_VALUE 2)} (:range edge)))))

(deftest static-local-pointer-ranges-come-only-from-the-validated-launch
  (let [checked (guarded-affine-effect-body)
        local (:id (first (filter #(and (instance? raster.compiler.ir.kernel_body.IndexBinding %)
                                        (= :local (:source %))) (:indices checked))))
        coordinate (kernel-body/expression :add local 128)
        load (kernel-body/->ScalarLoad (kernel-body/value 'validated-local-read :float)
                                       'input [coordinate] nil nil :cached)
        candidate (fn [extent]
                    (assoc checked :operations [load]
                           :launch (klaunch/spec {:workgroup-size [extent] :group-count [1]})))
        static (candidate 256)
        dynamic (candidate (klaunch/runtime-value 'width))
        oversized (candidate (inc (long Integer/MAX_VALUE)))]
    (is (= static (kernel-body/validate! static)))
    (doseq [body [dynamic oversized]]
      (is (= :kernel-body-index-precondition
             (try (kernel-body/validate! body) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "unproved local extents cannot certify an overflowing Int address prefix"))))

(deftest stencil-logical-long-geometry-widens-before-its-generated-prefixes
  (let [source '(let* [result (raster.par/stencil! du [u] 1 :dirichlet double i n
                              (clojure.core/aget u i))] result)
        typed (-> (typed-route/attempt source :double {'du :double 'u :double}
                                      {:scalar-types {'n :long}})
                  :program :equations first :algorithm)
        stencil (first (lower/lower-typed-stencil typed :cpu:0 :dtype :double))
        checked (:body (segstencil-body/schedule stencil
                          {:array-types {'du :double 'u :double}
                           :scalar-types {'n :long} :workgroup-size 256}))
        hardware (filter #(instance? raster.compiler.ir.kernel_body.IndexBinding %) (:indices checked))
        group (:id (first (filter #(= :group (:source %)) hardware)))
        local (:id (first (filter #(= :local (:source %)) hardware)))
        coordinate (:expression (first (filter #(instance? raster.compiler.ir.kernel_body.IndexCompute %)
                                               (:indices checked))))
        evaluate (fn evaluate [node values]
                   (cond (symbol? node) (get values node)
                         (integer? node) (bigint node)
                         (instance? raster.compiler.ir.kernel_body.IndexCast node)
                         (evaluate (:argument node) values)
                         (instance? raster.compiler.ir.kernel_body.IndexExpr node)
                         (apply ({:add +' :mul *'} (:op node))
                                (map #(evaluate % values) (:arguments node)))))
        casts (filter #(instance? raster.compiler.ir.kernel_body.IndexCast %)
                      (tree-seq coll? seq coordinate))
        geometry (klaunch/geometry {:workgroup-size [256] :group-count [8388609]})]
    (is (= :add (:op coordinate)))
    (is (= 3 (count casts)))
    (is (every? #(= :long (:dtype %)) casts))
    (is (= 2147483648N (evaluate coordinate {group 8388608 local 0})))
    (is (= 2147483903N (evaluate coordinate {group 8388608 local 255})))
    (is (= checked (kernel-body/validate! checked)))
    (is (= geometry (kernel-body/validate-launch-index-ranges! checked geometry)))
    (is (= :kernel-launch-index-range
           (try (kernel-body/validate-launch-index-ranges! checked
                  (klaunch/geometry {:workgroup-size [256] :group-count [2147483649]})) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest bound-source-index-cannot-bypass-a-reduction-owner-remapping
  (let [mapped (kernel-body/expression :add 'element-index 'offset)
        lowered (segred-body/lower-element-operations
                 '(aget fine (+ original-index offset))
                 {:index 'original-index :coordinate 'element-index :dtype :double
                  :arrays #{'fine} :array-types {'fine :double}
                  :scalars #{'offset} :scalar-types {'offset :long}
                  :element-binding-types {'original-index :long 'element-index :long}
                  :declared-result-dtype :double
                  :coordinate-lower (fn [coordinate]
                                      (when (= '(+ original-index offset) coordinate) mapped))})
        load (first (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                            (:operations lowered)))]
    (is (= [mapped] (:coordinates load)))
    (is (not-any? #{'original-index} (tree-seq coll? seq (:operations lowered))))
    (is (not-any? #(instance? raster.compiler.ir.kernel_body.ScalarCompute %)
                  (:operations lowered)))))

(deftest mixed-resident-and-ordinary-reduction-loads-retain-their-owner-adapter
  (let [lowered (segred-body/lower-element-operations
                 '(+ (aget resident 0) (aget fine original-index))
                 {:index 'original-index :coordinate 'element-index :dtype :double
                  :arrays #{'resident 'fine} :array-types {'resident :double 'fine :double}
                  :scalars #{} :scalar-types {}
                  :element-binding-types {'original-index :long 'element-index :long}
                  :declared-result-dtype :double
                  :lower-load-index (fn [array coordinate _]
                                      (when (and (= array 'resident) (= coordinate 0))
                                        (kernel-body/index-cast 0 :long :exact)))
                  :coordinate-lower (fn [coordinate]
                                      (when (= coordinate 'original-index) 'element-index))})
        loads (filter #(instance? raster.compiler.ir.kernel_body.ScalarLoad %)
                      (:operations lowered))]
    (is (= ['resident 'fine] (mapv :buffer loads)))
    (is (= [(kernel-body/index-cast 0 :long :exact)] (:coordinates (first loads))))
    (is (= ['element-index] (:coordinates (second loads))))
    (is (not-any? #{'original-index} (tree-seq coll? seq (:operations lowered))))))

(deftest entered-strided-loops-preserve-the-actual-widened-abi-upper-range
  (let [info-var (ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!)
        original @info-var
        run (fn [width-type constant expected]
              (let [seen (atom nil)
                    width 'typed-entry-width
                    index 'typed-entry-index
                    parameter? (nil? constant)
                    loop (kernel-body/->ForLoop
                           (kernel-body/value index :long)
                           (kernel-body/index-cast 'typed-entry-lane :long :exact)
                           (kernel-body/index-cast width :long :exact) 1024 []
                           [(kernel-body/->ScalarCompute
                              (kernel-body/value 'typed-entry-copy :long)
                              (kernel-body/scalar-expression :+ :long
                                [index (kernel-body/literal 0 :long)]
                                {:overflow :no-overflow
                                 :proof (assoc expected :kind :typed-scalar-range)}))
                            (kernel-body/->Yield [])] [] {})
                    spec {:id :typed-entered-upper
                          :parameters (if parameter?
                                        [(kernel-body/->KernelParameter width :scalar width-type [] nil nil :parameter)] [])
                          :indices [(kernel-body/->IndexBinding 'typed-entry-lane :local 0)]
                          :launch (klaunch/spec {:workgroup-size [1024] :group-count [1]})
                          :operations (cond-> []
                                        (not parameter?)
                                        (conj (kernel-body/->ScalarCompute
                                                (kernel-body/value width :long)
                                                (kernel-body/cast-expression
                                                  (kernel-body/literal constant :long) :long :exact :exact)))
                                        true (conj loop))}
                    checked (with-redefs-fn
                              {info-var (fn [expression values]
                                          (when (contains? values index)
                                            (reset! seen (get values index)))
                                          (original expression values))}
                              #(kernel-body/make spec))]
                (is (kernel-body/kernel-body? checked))
                (is (= expected (:range @seen)))
                @seen))]
    (is (= 'typed-entry-width (get-in (run :int nil {:lower 0 :upper (dec (long Integer/MAX_VALUE))})
                                             [:entered-loop-index :upper])))
    (is (= 'typed-entry-width (get-in (run :long nil {:lower 0 :upper (dec (bigint Long/MAX_VALUE))})
                                             [:entered-loop-index :upper])))
    (doseq [upper [0 -1]]
      (is (nil? (:entered-loop-index (run :long upper {:lower (min 0 upper) :upper 1023})))))))


(deftest scan-generated-global-prefixes-retain-long-width-before-arithmetic
  (let [emitted (emitted-scan-graph
                 '(raster.par/scan out acc 0.0 i n double (+ acc (aget values i))))
        evaluate (fn evaluate [node values]
                   (cond (symbol? node) (get values node)
                         (integer? node) (bigint node)
                         (instance? raster.compiler.ir.kernel_body.IndexCast node)
                         (evaluate (:argument node) values)
                         (instance? raster.compiler.ir.kernel_body.IndexExpr node)
                         (apply ({:add +' :mul *'} (:op node))
                                (map #(evaluate % values) (:arguments node)))))]
    (doseq [artifact [(get-in emitted [:nodes 0 :operation])
                      (get-in emitted [:nodes 2 :operation])]]
      (let [checked (get-in artifact [:attributes :kernel-body])
            width (first (get-in checked [:launch :workgroup-size]))
            coordinate (:expression (first (filter #(= 'scan-index (:id %)) (:indices checked))))
            casts (filter #(instance? raster.compiler.ir.kernel_body.IndexCast %)
                          (tree-seq coll? seq coordinate))
            group (inc (quot Integer/MAX_VALUE width))
            expected (*' group width)
            geometry (klaunch/geometry {:workgroup-size [width] :group-count [(inc group)]})]
        (is (= 3 (count casts)))
        (is (every? #(and (= :long (:dtype %)) (= :exact (:overflow %))) casts))
        (is (> expected Integer/MAX_VALUE))
        (is (= expected (evaluate coordinate {'scan-group group 'scan-lane 0})))
        (is (= (+ expected (dec width))
               (evaluate coordinate {'scan-group group 'scan-lane (dec width)})))
        (is (= :int (:dtype (first (filter #(= '_n_bound (:id %)) (:parameters checked))))))
        (is (= checked (kernel-body/validate! checked)))
        (is (= geometry (kernel-body/validate-launch-index-ranges! checked geometry)))
        (is (= :kernel-launch-index-range
               (try (kernel-body/validate-launch-index-ranges! checked
                      (klaunch/geometry {:workgroup-size [width] :group-count [2147483649]})) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest ordinary-quotient-ranges-use-only-closed-verified-nonnegative-origins
  (let [address-var (ns-resolve 'raster.compiler.ir.kernel-body 'address-expression-uniformity)
        original @address-var
        run (fn [numerator divisor external?]
              (let [seen (atom nil)
                    spec {:id :ordinary-quotient-origin
                          :parameters (cond-> [(kernel-body/->KernelParameter
                                                 'out :output :long [1] :global
                                                 {:kind :row-major :rank 1 :shape [1] :perm [0] :dtype :long}
                                                 :result)]
                                        external? (conj (assoc (kernel-body/->KernelParameter
                                                                 'external :scalar :int [] nil nil :parameter)
                                                               :nonnegative? true))
                                        (= divisor :unknown) (conj (kernel-body/->KernelParameter
                                                                    'divisor :scalar :long [] nil nil :parameter)))
                          :indices [(kernel-body/->IndexBinding 'origin-group :group 0)
                                    (kernel-body/->IndexCompute 'origin-numerator numerator)
                                    (kernel-body/->IndexCompute 'origin-quotient
                                      (kernel-body/expression :floor-div 'origin-numerator
                                        (if (= divisor :unknown) 'divisor
                                            (kernel-body/index-cast divisor :long :exact))))]
                          :operations [(kernel-body/->ScalarStore 'out
                                         [(kernel-body/expression :sub 'origin-quotient
                                            (kernel-body/index-cast 1 :long :exact))]
                                         (kernel-body/literal 0 :long) nil)]
                          :launch (klaunch/spec {:workgroup-size [1] :group-count [1]})}
                    outcome (try (with-redefs-fn
                                   {address-var (fn [expression values]
                                                  (reset! seen (get values 'origin-quotient))
                                                  (original expression values))}
                                   #(kernel-body/make spec))
                                 :accepted
                                 (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))]
                {:outcome outcome :info @seen}))
        widened (kernel-body/expression :mul
                  (kernel-body/index-cast 'origin-group :long :exact)
                  (kernel-body/index-cast 256 :long :exact))
        positive (run widened 16 false)]
    (is (= :accepted (:outcome positive)))
    (is (= {:lower 0 :upper (quot (*' Integer/MAX_VALUE 256) 16)}
           (get-in positive [:info :range])))
    (doseq [divisor [0 -1 :unknown]]
      (is (= :kernel-body-index-precondition (:outcome (run widened divisor false)))))
    (is (= :kernel-body-index-precondition
           (:outcome (run (kernel-body/index-cast 'external :long :exact) 16 true)))
        "a producer sign field on an ABI parameter supplies no hardware proof")
    (is (= :kernel-body-index-precondition
           (:outcome (run (kernel-body/index-cast
                            (kernel-body/expression :mul 'origin-group 256) :long :exact) 16 false)))
        "a widening alias cannot erase the evaluated overflowing Int prefix")))

(deftest varying-map-guard-reconstructs-the-actual-modulo-domain
  (let [operation (segop/->SegMap
                   992 (segop/make-seg-space 'row 'n) (segop/->SegLevel :thread :virtual)
                   '(+ (clojure.core/aget x row)
                       (clojure.core/aget bias (mod row features)))
                   nil #{'x 'bias} #{'output} #{'features}
                   (segop/->KernelGrid 1 256 0) :float 'output nil)
        options {:scalar-types {'n :long 'features :long}
                 :array-types {'x :float 'bias :float 'output :float}}
        scheduled (segmap-body/schedule operation options)
        checked (:body scheduled)
        conditions (kernel-body/required-preconditions checked)
        guard (first (:operations checked))
        index (:id (last (:indices checked)))
        region {:parameters (:parameters checked) :indices (:indices checked)
                :launch (:launch checked) :masks (:masks checked) :guard guard
                :index-definitions [] :required-preconditions conditions}
        reason-of (fn [f] (try (f) :accepted
                              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
        bindings (fn [n features] (fn [id] (case id _n_bound n features features)))]
    (is (= checked (kernel-body/validate! checked)))
    (is (= conditions (:preconditions scheduled)))
    (is (= '_n_bound (get-in (kernel-body/varying-guard-index-region-info region)
                             [:values index :active-upper])))
    (is (= :kernel-body-index-region-guard
           (reason-of #(kernel-body/scheduled-index-region-info region)))
        "cooperative entry still requires actual workgroup uniformity")
    (doseq [candidate [(dissoc region :guard) (assoc region :masks [])
                       (assoc region :if-region (kernel-body/->IfRegion 'fake [] [] []))]]
      (is (= :kernel-body-index-region-guard
             (reason-of #(kernel-body/varying-guard-index-region-info candidate)))))
    (let [unknown-mask (assoc-in region [:masks 0 :predicates]
                                 [(kernel-body/predicate :lt 'unknown-index
                                    (kernel-body/index-cast '_n_bound :long :exact))])]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"kernel mask"
                           (kernel-body/varying-guard-index-region-info unknown-mask))))
    (doseq [[n f] [[1 1] [0 0] [0 Long/MIN_VALUE] [0 Long/MAX_VALUE]]]
      (is (true? (precondition/check! conditions (bindings n f)))))
    (doseq [[n f] [[1 0] [1 Long/MIN_VALUE] [-1 1]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (precondition/check! conditions (bindings n f)))))
    (doseq [condition conditions]
      (is (= :kernel-body-index-precondition
             (reason-of #(kernel-body/validate!
                          (assoc-in checked [:attributes :required-preconditions]
                                    (vec (remove #{condition} conditions))))))))
    (let [barrier (kernel-body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                                  (kernel-body/full-participation))]
      (is (= :kernel-body-divergent-workgroup-barrier
             (reason-of #(kernel-body/validate!
                          (assoc-in checked [:operations 0 :operations] [barrier]))))))
    (is (= :kernel-body-index-precondition
           (reason-of #(kernel-body/validate! (assoc checked :operations (:operations guard)))))
        "an unguarded evaluated modulo cannot borrow the later lexical domain")))


(deftest initial-schedule-scalar-proofs-replay-real-geometry-and-restore-scope
  (let [launch (klaunch/spec {:workgroup-size [256] :group-count [1]})
        indices [(kernel-body/->IndexBinding 'g :group 0)
                 (kernel-body/->IndexBinding 'lane :local 0)
                 (kernel-body/->IndexCompute
                  'idx (kernel-body/expression :add
                         (kernel-body/expression :mul
                           (kernel-body/index-cast 'g :long :exact)
                           (kernel-body/index-cast 256 :long :exact))
                         (kernel-body/index-cast 'lane :long :exact)))]
        lowerer (scalar-body/make-lowerer
                 {:arrays #{} :array-types {} :scalar-types {'idx :long}
                  :index-scope #{'idx} :lower-index (fn [x _] x)
                  :decline! (fn [r m d] (throw (ex-info m (assoc d :missing-rule r))))})
        compute #((:lower lowerer) '(+ idx (long 1)) :long {'idx :long})
        inside ((:with-initial-index-region lowerer)
                {:parameters [] :indices indices :launch launch} compute)
        outside (compute)
        expression (fn [lowered]
                     (:expression (first (filter #(= (:result lowered) (get-in % [:result :id]))
                                                 (:operations lowered)))))
        checked (kernel-body/make
                 {:id :real-initial-scope :indices indices :launch launch
                  :operations (:operations inside)})]
    (is (= :no-overflow (get-in (expression inside) [:options :overflow])))
    (is (= :trap (get-in (expression outside) [:options :overflow]))
        "initial facts cannot escape into later source lowering")
    (is (= checked (kernel-body/validate! checked)))
    (let [opaque ((:with-initial-index-region lowerer)
                  {:parameters [(kernel-body/->KernelParameter 'idx :scalar :long [] nil nil :parameter)]
                   :indices [] :launch launch} compute)]
      (is (= :trap (get-in (expression opaque) [:options :overflow]))
          "a full Long ABI scalar is not a hardware-coordinate proof"))
    (doseq [bad [[(kernel-body/->IndexBinding 'g :unknown 0)]
                 [(kernel-body/->IndexCompute 'idx (kernel-body/expression :add 'undefined 1))]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (kernel-body/initial-scheduled-index-info
                    {:parameters [] :indices bad :launch launch}))))))


(deftest initial-schedule-keeps-unsafe-prefix-ancestry-and-restores-after-throw
  (let [launch (klaunch/spec {:workgroup-size [256] :group-count [1]})
        bad-indices [(kernel-body/->IndexBinding 'g :group 0)
                     (kernel-body/->IndexCompute
                      'idx (kernel-body/index-cast
                            (kernel-body/expression :mul 'g 256) :long :exact))]
        info (kernel-body/initial-scheduled-index-info
              {:parameters [] :indices bad-indices :launch launch})
        lowerer (scalar-body/make-lowerer
                 {:arrays #{} :array-types {} :scalar-types {'idx :long}
                  :index-scope #{'idx} :lower-index (fn [x _] x)
                  :decline! (fn [r m d] (throw (ex-info m (assoc d :missing-rule r))))})
        lowered ((:with-initial-index-region lowerer)
                 {:parameters [] :indices bad-indices :launch launch}
                 #((:lower lowerer) '(+ idx (long 1)) :long {'idx :long}))]
    (is (true? (get-in info [:values 'idx :unresolved-index-address-origin?])))
    (try
      (kernel-body/make
       {:id :unsafe-initial-prefix :indices bad-indices :launch launch
        :parameters [(kernel-body/->KernelParameter 'x :input :float [1] :global (layout/row-major [1] :float) :operand)]
        :operations (conj (vec (:operations lowered))
                          (kernel-body/->ScalarLoad (kernel-body/value 'loaded :float)
                                                    'x [(:result lowered)] nil nil :cached))})
      (is false "a final Long cast cannot certify an already unsafe Int address prefix")
      (catch clojure.lang.ExceptionInfo e
        (is (= :kernel-body-index-precondition (:reason (ex-data e)))))))
  (let [launch (klaunch/spec {:workgroup-size [256] :group-count [1]})
        indices [(kernel-body/->IndexBinding 'lane :local 0)
                 (kernel-body/->IndexCompute 'idx (kernel-body/index-cast 'lane :long :exact))]
        lowerer (scalar-body/make-lowerer
                 {:arrays #{} :array-types {} :scalar-types {'idx :long}
                  :index-scope #{'idx} :lower-index (fn [x _] x)
                  :decline! (fn [r m d] (throw (ex-info m (assoc d :missing-rule r))))})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scope callback"
          ((:with-initial-index-region lowerer)
           {:parameters [] :indices indices :launch launch}
           #(throw (ex-info "scope callback" {})))))
    (let [outside ((:lower lowerer) '(+ idx (long 1)) :long {'idx :long})]
      (let [operation (first (filter #(= (:result outside) (get-in % [:result :id]))
                                     (:operations outside)))]
        (is (= :+ (get-in operation [:expression :op])))
        (is (= :trap (get-in operation [:expression :options :overflow])))))))

(deftest independently-bounded-products-retain-every-executed-child-cap
  (let [parameters (mapv #(kernel-body/->KernelParameter % :scalar :long [] nil nil :parameter)
                         '[N rows head-dim n-q])
        indices [(kernel-body/->IndexBinding 'raw-group :group 0)
                 (kernel-body/->IndexCompute 'idx (kernel-body/index-cast 'raw-group :long :exact))]
        launch (klaunch/spec {:workgroup-size [8] :group-count [1]})
        condition (kernel-body/->ScalarCompute (kernel-body/value 'active :predicate)
                    (kernel-body/scalar-expression :lt :predicate ['idx 'N]))
        definitions [(kernel-body/->IndexCompute 'remainder
                        (kernel-body/expression :mod
                          (kernel-body/expression :floor-div 'idx 'head-dim) 'rows))]
        branch (kernel-body/->IfRegion 'active definitions [(kernel-body/->Yield [])] [])
        domains (vec (distinct (concat (kernel-body/quotient-domain-preconditions 'N 'head-dim)
                                      (kernel-body/quotient-domain-preconditions 'N 'rows))))
        region {:parameters parameters :indices indices :launch launch
                :condition-definition condition :if-region branch :index-definitions definitions
                :required-preconditions domains}
        lowerer (scalar-body/make-lowerer
                 {:arrays #{} :lower-index (fn [expression _] expression)
                  :scalar-types (zipmap '[N rows head-dim n-q idx remainder] (repeat :long))
                  :id-prefix "independent-product" :decline! (fn [rule message data]
                                                              (throw (ex-info message (assoc data :rule rule))))})
        lowered ((:with-index-region lowerer) region
                 #((:lower lowerer) '(* remainder (* n-q head-dim)) :long {}))
        requirements ((:required-preconditions lowerer))
        kernel (kernel-body/make {:id :independent-nonnegative-product :parameters parameters
                          :indices indices :launch launch :attributes {:required-preconditions requirements}
                          :operations [condition (assoc branch :then-operations
                                                  (into definitions (concat (:operations lowered)
                                                                            [(kernel-body/->Yield [])])))]})
        product (first (filter #(= (:result lowered) (get-in % [:result :id]))
                               (:operations lowered)))]
    (is (= :* (get-in product [:expression :op])))
    (is (= :no-overflow (get-in product [:expression :options :overflow])))
    (is (kernel-body/kernel-body? (kernel-body/validate! kernel)))
    (is (some? (:source (scalar-emitter/emit-scalar-module "independent_product" kernel :opencl-portable))))
    (is (true? (precondition/check! requirements {'N 3 'rows 3 'head-dim 1 'n-q 2})))
    ;; Child n-q*head-dim fits at MAX, but the actual remainder may be two.
    ;; The separately transported parent cap must reject 2*MAX before execution.
    (is (thrown? ArithmeticException
                 (precondition/check! requirements {'N 3 'rows 3 'head-dim 1 'n-q Long/MAX_VALUE})))
    (is (thrown? ArithmeticException
                 (precondition/check! requirements {'N 1 'rows 1 'head-dim 2 'n-q Long/MAX_VALUE}))
        "a zero parent remainder cannot erase the executed overflowing child product")
    (is (true? (precondition/check! requirements
                                  {'N 0 'rows Long/MIN_VALUE 'head-dim 0 'n-q Long/MIN_VALUE})))
    (doseq [binding [{'N 3 'rows 0 'head-dim 1 'n-q 2}
                     {'N 3 'rows 3 'head-dim 0 'n-q 2}
                     {'N 3 'rows 3 'head-dim 1 'n-q -1}]]
      (is (thrown? clojure.lang.ExceptionInfo (precondition/check! requirements binding))))
    (doseq [requirement requirements]
      (let [outcome (try
                      (kernel-body/validate! (assoc-in kernel [:attributes :required-preconditions]
                                               (vec (remove #{requirement} requirements))))
                      {:outcome :accepted}
                      (catch clojure.lang.ExceptionInfo error
                        {:outcome :rejected :data (ex-data error)}))]
        (is (= :rejected (:outcome outcome)) (pr-str outcome))
        (is (contains? #{:kernel-body-index-precondition :kernel-body-intrinsic-overflow
                         :kernel-body-intrinsic-overflow-proof} (get-in outcome [:data :reason]))
            (pr-str outcome))))
    (let [checked-branch (second (:operations kernel))
          reject-proof (fn [changed]
                         (let [outcome (try (kernel-body/validate! changed) {:outcome :accepted}
                                           (catch clojure.lang.ExceptionInfo error
                                             {:outcome :rejected :data (ex-data error)}))]
                           (is (= :rejected (:outcome outcome)) (pr-str outcome))
                           (is (contains? #{:kernel-body-index-precondition :kernel-body-intrinsic-overflow
                                            :kernel-body-intrinsic-overflow-proof}
                                          (get-in outcome [:data :reason])) (pr-str outcome))))]
      (reject-proof (assoc kernel :operations (vec (butlast (:then-operations checked-branch)))))
      (reject-proof (assoc-in kernel [:operations 1]
                             (assoc checked-branch :then-operations [(kernel-body/->Yield [])]
                                                   :else-operations (:then-operations checked-branch))))
      (reject-proof (assoc-in kernel [:indices 1 :expression]
                             (kernel-body/index-cast
                              (kernel-body/expression :mul 'raw-group 256) :long :exact))))
    ;; The mixed-product rule must not equate separately reconstructed domains.
    ;; This tests the rule only; neither synthetic environment is an executable certificate.
    (let [values (:values (kernel-body/scheduled-index-region-info region))
          scalar-info (ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!)
          uniform (first (filter #(= #{'n-q 'head-dim} (set (get-in % [:expression :arguments])))
                                 (:operations lowered)))
          other-domains (mapv #(update % :expression (fn [expression]
                                                       (walk/postwalk-replace {'N 'M} expression))) domains)
          other-region (-> region
                           (update :parameters conj (kernel-body/->KernelParameter 'M :scalar :long [] nil nil :parameter))
                           (assoc :required-preconditions other-domains)
                           (assoc-in [:condition-definition :expression :arguments] ['idx 'M]))
          other-values (:values (kernel-body/scheduled-index-region-info other-region))
          child-transfer (kernel-body/conditional-integer-transfer (:expression uniform) other-values)
          other-values (with-meta other-values
                         (assoc (meta other-values) :required-preconditions
                                (vec (distinct (concat other-domains (:required-preconditions child-transfer))))))
          other-factor (scalar-info (:expression uniform) other-values)
          negative (scalar-info (kernel-body/cast-expression (kernel-body/literal -1 :long) :long :exact :exact) values)
          expression (kernel-body/scalar-expression :* :long ['remainder 'other-factor] {})]
      (is (some? uniform))
      (is (= 'M (get-in other-factor [:conditional-integer :bound])))
      (is (nil? (kernel-body/conditional-integer-transfer expression (assoc values 'other-factor other-factor))))
      (is (nil? (kernel-body/conditional-integer-transfer expression (assoc values 'other-factor negative)))))))
