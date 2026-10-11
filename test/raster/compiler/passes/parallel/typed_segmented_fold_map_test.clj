(ns raster.compiler.passes.parallel.typed-segmented-fold-map-test
  (:require [clojure.string :as str]
            [clojure.set :as set]
            [clojure.walk :as walk]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.extent-expression :as extent-expression]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.passes.parallel.segfoldmap-body :as fold-body]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par :as par]))

(def ^:private source
  '(let* [effect
          (raster.par/segmented-fold-map!
           [out] [[segment nsegments]] index width
           [[maximum -1.0E38 :float width
             (clojure.core/max
              maximum
              (clojure.core/aget values
                                 (clojure.core/+ (clojure.core/* segment width) index)))]
            [denominator 0.0 :float width
             (clojure.core/+
              denominator
              (Math/exp
               (clojure.core/-
                (clojure.core/aget values
                                   (clojure.core/+ (clojure.core/* segment width) index))
                maximum)))]]
           [(clojure.core//
             (Math/exp
              (clojure.core/-
               (clojure.core/aget values
                                  (clojure.core/+ (clojure.core/* segment width) index))
               maximum))
             denominator)])]
         effect))

(def ^:private certified-source
  '(let* [effect
          (raster.par/segmented-fold-map!
           [out] [[row rows]] index width
           [[sum 0.0 :float width
             (clojure.core/+
              sum
              (clojure.core/aget values
                                 (clojure.core/+ (clojure.core/* row width) index)))
             {:association :implementation-defined}]]
           [(clojure.core/float
             (clojure.core/*
              sum
              (clojure.core/aget values
                                 (clojure.core/+ (clojure.core/* row width) index))))])]
         effect))

(def ^:private certified-maximum-source
  '(let* [effect
          (raster.par/segmented-fold-map!
           [out] [[row rows]] index 1
           [[maximum Float/NEGATIVE_INFINITY :float width
             (clojure.core/max
              maximum
              (clojure.core/aget values
                                 (clojure.core/+ (clojure.core/* row width) index)))
             {:association :implementation-defined}]]
           [(clojure.core/float maximum)])]
         effect))

(def ^:private certified-minimum-source
  (walk/postwalk-replace
   {'clojure.core/max 'clojure.core/min
    'Float/NEGATIVE_INFINITY 'Float/POSITIVE_INFINITY}
   certified-maximum-source))

(def ^:private certified-multi-fold-source
  '(let* [effect
          (raster.par/segmented-fold-map!
           [sum-out maximum-out] [[row rows]] index 1
           [[sum 0 :int width
             (clojure.core/unchecked-add-int
              sum
              (clojure.core/aget counts
                                 (clojure.core/+ (clojure.core/* row width) index)))
             {:association :implementation-defined}]
            [maximum Float/NEGATIVE_INFINITY :float width
             (clojure.core/max
              maximum
              (clojure.core/+
               (clojure.core/aget values
                                  (clojure.core/+ (clojure.core/* row width) index))
               (clojure.core/float sum)))
             {:association :implementation-defined}]]
           [(clojure.core/float sum) maximum])]
         effect))

(defn- scheduled-operation []
  (let [program (:program
                 (route/attempt source :float {'values :float 'out :float}
                                {:scalar-types {'nsegments :int 'width :int}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'nsegments :int 'width :int}}))]
    (first (:operations (first (:equations scheduled))))))

(defn- certified-operation []
  (let [program (:program
                 (route/attempt certified-source :float {'values :float 'out :float}
                                {:scalar-types {'rows :long 'width :long}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'rows :long 'width :long}}))]
    (first (:operations (first (:equations scheduled))))))

(defn- certified-maximum-operation []
  (let [program (:program
                 (route/attempt certified-maximum-source :float
                                {'values :float 'out :float}
                                {:scalar-types {'rows :long 'width :long}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'rows :long 'width :long}}))]
    (first (:operations (first (:equations scheduled))))))

(defn- certified-minimum-operation []
  (let [program (:program
                 (route/attempt certified-minimum-source :float
                                {'values :float 'out :float}
                                {:scalar-types {'rows :long 'width :long}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'rows :long 'width :long}}))]
    (first (:operations (first (:equations scheduled))))))

(defn- certified-multi-fold-operation []
  (let [array-types {'counts :int 'values :float
                     'sum-out :float 'maximum-out :float}
        scalar-types {'rows :long 'width :long}
        program (:program
                 (route/attempt certified-multi-fold-source :float array-types
                                {:scalar-types scalar-types}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types array-types :scalar-types scalar-types}))]
    (first (:operations (first (:equations scheduled))))))

(defn- decline-rule
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (when (fold-body/declined? exception)
        (:missing-rule (ex-data exception))))))

(defn- reason-of
  [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

(defn- operation-tree
  [operations]
  (mapcat (fn [operation]
            (cons operation
                  (cond
                    (or (instance? raster.compiler.ir.kernel_body.ForLoop operation)
                        (instance? raster.compiler.ir.kernel_body.Guard operation))
                    (operation-tree (:operations operation))
                    (instance? raster.compiler.ir.kernel_body.IfRegion operation)
                    (concat (operation-tree (:then-operations operation))
                            (operation-tree (:else-operations operation)))
                    :else [])))
          operations))

(deftest launch-and-kernel-index-maximum-have-one-canonical-storage-form
  (is (extent-expression/equivalent? (launch/maximum 'n 1)
                                     (body/expression :max 'n 1))))

(deftest interpreted-segmented-fold-map-preserves-dependent-fold-order
  (let [values (float-array [1.0 3.0, 2.0 2.0])
        out (float-array 4)]
    (par/segmented-fold-map!
     [out] [[segment 2]] index 2
     [[maximum -1.0E38 :float 2
       (max maximum (aget values (+ (* segment 2) index)))]
      [denominator 0.0 :float 2
       (+ denominator (Math/exp (- (aget values (+ (* segment 2) index)) maximum)))]]
     [(/ (Math/exp (- (aget values (+ (* segment 2) index)) maximum)) denominator)])
    (is (< (Math/abs (- (aget out 0) 0.11920292)) 1.0e-6))
    (is (< (Math/abs (- (aget out 1) 0.8807971)) 1.0e-6))
    (is (< (Math/abs (- (aget out 2) 0.5)) 1.0e-6))
    (is (< (Math/abs (- (aget out 3) 0.5)) 1.0e-6))))

(deftest fold-map-results-retain-distinct-destination-storage-types
  (let [source '(let* [effect (raster.par/segmented-fold-map!
                              [small wide] [[row 1]] j 1
                              [[sum 0.0 :double 3 (+ sum (double (aget values j)))]]
                              [sum sum])] effect)
        program (frontend/form->program source {:dtype :double
                                                :array-types {'values :float 'small :float 'wide :double}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)]
    (is (= [:float :double] (get-in operation [:attributes :dtypes])))
    (is (= :double (get-in operation [:folds 0 :attributes :dtype])))
    (is (= :float (:dtype (get-in (dialect/facts program) [:values (first (nth equation 2))]))))
    (is (= :double (:dtype (get-in (dialect/facts program) [:values (second (nth equation 2))]))))
    (let [projected (projection/segmented-fold-map-form program equation)
          run (eval (list 'fn '[values small wide] projected))
          source-run (eval (list 'fn '[^floats values ^floats small ^doubles wide] source))
          values (float-array [16777216.0 1.0 -16777216.0])
          small (float-array 1)
          wide (double-array 1)]
      (run values small wide)
      (is (= [1.0] (vec small)) "only the completed double carry is narrowed")
      (is (= [1.0] (vec wide)) "the second output keeps its independent storage type")
      (doseq [input [values (float-array [16777216.0 1.0 0.0])]]
        (let [source-small (float-array 1)
              source-wide (double-array 1)]
          (source-run input source-small source-wide)
          (run input small wide)
          (is (= (vec source-small) (vec small)))
          (is (= (vec source-wide) (vec wide))))))))

(deftest dependent-folds-are-first-class-typed-soac
  (let [program (frontend/form->program
                 source {:dtype :float :array-types {'values :float 'out :float}
                         :scalar-types {'nsegments :int 'width :int}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)]
    (is (= 'segmented-fold-map (:kind operation)))
    (is (= 2 (count (:folds operation))))
    (is (= '[nsegments values width] (:inputs (dialect/facts program))))
    (is (= '[out] (dialect/physical-results program equation)))
    (is (= :int (:dtype (get-in (dialect/facts program) [:values 'width])))
        "launch dimensions retain their declared ABI representation")))

(deftest reassociation-is-an-explicit-derived-fold-contract
  (let [options {:dtype :float :array-types {'values :float 'out :float}
                 :scalar-types {'rows :long 'width :long}}
        program (frontend/form->program certified-source options)
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)
        fold (first (:folds operation))
        attributes (:attributes fold)
        projected (projection/segmented-fold-map-form program equation)]
    (is (= :per-fold (get-in operation [:attributes :association])))
    (is (= :implementation-defined (:association attributes)))
    (is (scan/associative-scan? (:algebra attributes)))
    (is (= {:association :implementation-defined}
           (peek (first (nth projected 5))))
        "the explicit permission survives the temporary host projection")
    (is (= program (dialect/validate! program)))
    (let [declared (:algebra attributes)
          forged (assoc declared :combine 'clojure.core/*)
          corrupted (walk/postwalk-replace {declared forged} program)]
      (is (= :typed-soac-segmented-fold-map-certificate
             (reason-of #(dialect/validate! corrupted)))
          "a serialized certificate is checked against the retained scalar region"))
    (let [values (float-array [1.0 2.0 3.0 4.0])
          out (float-array 4)
          run (eval (list 'fn '[values out rows width] projected))]
      (run values out 2 2)
      (is (= [3.0 6.0 21.0 28.0] (vec out)))
      (is (= [1.0 2.0 3.0 4.0] (vec values))))))

(deftest reassociation-request-is-not-a-user-supplied-proof
  (let [invalid (walk/postwalk-replace
                 {'clojure.core/+ 'clojure.core/-}
                 certified-source)]
    (is (= :segmented-fold-map-not-associative
           (reason-of
            #(frontend/form->program
              invalid {:dtype :float :array-types {'values :float 'out :float}
                       :scalar-types {'rows :long 'width :long}}))))))

(deftest certified-fold-map-selects-one-workgroup-per-segment
  (let [operation (certified-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'rows :long 'width :long}}
        {:keys [kernel-body workgroup-size]} (fold-body/lower operation options)
        scheduled (fold-body/schedule operation options)
        loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                      (operation-tree (:operations kernel-body)))
        barriers (filter #(instance? raster.compiler.ir.kernel_body.WorkgroupBarrier %)
                         (operation-tree (:operations kernel-body)))]
    (is (= :one-workgroup-per-segment (get-in kernel-body [:schedule :strategy])))
    (is (= :certified (get-in kernel-body [:schedule :association])))
    (is (= [workgroup-size] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [(launch/runtime-value 'rows)]
           (get-in kernel-body [:launch :group-count])))
    (is (= (* workgroup-size 4)
           (get-in kernel-body [:launch :shared-memory-bytes])))
    (is (= [:cooperative-lane-fold :cooperative-final-map]
           (mapv #(get-in % [:attributes :role]) loops)))
    (is (= workgroup-size (:step (first loops))))
    (is (= workgroup-size (:step (second loops))))
    (is (= (+ 2 (Long/numberOfTrailingZeros (long workgroup-size)))
           (count barriers))
        "the tree has one barrier per level plus publication and post-result reuse")
    (is (= :one-workgroup-per-segment
           (get-in scheduled [:legality :segment-parallelism])))
    (is (= :certified-workgroup-tree (get-in scheduled [:numerics :policy])))
    (is (= :implementation-defined (get-in scheduled [:numerics :reassociation])))))

(deftest one-cooperative-body-emits-for-all-c-family-targets
  (let [operation (certified-operation)]
    (doseq [[target expected]
            [[:opencl-portable :opencl-c]
             [:cuda :cuda-c]
             [:hip :hip-cpp]]]
      (testing (name target)
        (let [artifact (segop-opencl/generate-segfoldmap-kernel
                        operation :target-dialect target
                        :array-types {'values :float 'out :float}
                        :scalar-types {'rows :long 'width :long})]
          (is (= expected (:target artifact)))
          (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
          (is (= :one-workgroup-per-segment
                 (get-in artifact [:attributes :kernel-body :schedule :strategy])))
          (is (str/includes? (:source artifact) "foldmap_workgroup_scratch")))))))

(deftest multiple-certified-folds-share-one-cooperative-segment-schedule
  (let [operation (certified-multi-fold-operation)
        options {:array-types {'counts :int 'values :float
                               'sum-out :float 'maximum-out :float}
                 :scalar-types {'rows :long 'width :long}}
        {:keys [kernel-body operators workgroup-size]} (fold-body/lower operation options)
        scheduled (fold-body/schedule operation options)
        loops (filter #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                      (operation-tree (:operations kernel-body)))
        allocations (:allocations kernel-body)]
    (is (= [:+ :max] operators))
    (is (= [:+ :max] (get-in kernel-body [:schedule :reduction-operators])))
    (is (= 2 (get-in kernel-body [:schedule :fold-count])))
    (is (= [:int :float] (mapv :dtype allocations)))
    (is (= (* workgroup-size (+ 4 4))
           (get-in kernel-body [:launch :shared-memory-bytes])))
    (is (= [:cooperative-lane-fold :cooperative-lane-fold :cooperative-final-map]
           (mapv #(get-in % [:attributes :role]) loops)))
    (is (= [0 1] (mapv #(get-in % [:attributes :fold]) (butlast loops))))
    (is (= [:int :float] (get-in scheduled [:numerics :accumulator-dtypes])))
    (is (= :wrap (get-in scheduled [:numerics :accumulators 0 :overflow])))
    (is (= '[counts values sum-out maximum-out]
           (get-in scheduled [:attributes :array-params])))
    (is (= [:+ :max] (get-in scheduled [:attributes :reduction-operators])))))

(deftest checked-integral-addition-is-not-silently-reassociated
  (let [checked-source (walk/postwalk-replace
                        {'clojure.core/unchecked-add-int 'clojure.core/+}
                        certified-multi-fold-source)
        array-types {'counts :int 'values :float
                     'sum-out :float 'maximum-out :float}
        scalar-types {'rows :long 'width :long}
        program (:program (route/attempt checked-source :float array-types
                                         {:scalar-types scalar-types}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types array-types :scalar-types scalar-types}))
        operation (first (:operations (first (:equations scheduled))))]
    (is (= :integral-monoid-overflow
           (decline-rule #(fold-body/lower operation
                                           {:array-types array-types
                                            :scalar-types scalar-types}))))))

(deftest static-multi-fold-extent-caps-the-cooperative-workgroup
  (let [operation (update (certified-multi-fold-operation) :folds
                          #(mapv (fn [fold] (assoc fold :extent 8)) %))
        {:keys [kernel-body workgroup-size]}
        (fold-body/lower operation
                         {:array-types {'counts :int 'values :float
                                        'sum-out :float 'maximum-out :float}
                          :scalar-types {'rows :long 'width :long}})]
    (is (= 8 workgroup-size))
    (is (= [8 8] (mapv (comp first :shape) (:allocations kernel-body))))
    (is (= 64 (get-in kernel-body [:launch :shared-memory-bytes])))))

(deftest multiple-certified-folds-retain-lexical-dependence-in-projection
  (let [program (frontend/form->program
                 certified-multi-fold-source
                 {:dtype :float
                  :array-types {'counts :int 'values :float
                                'sum-out :float 'maximum-out :float}
                  :scalar-types {'rows :long 'width :long}})
        projected (projection/segmented-fold-map-form
                   program (first (dialect/equations program)))
        [_ outputs _ _ _ folds results] projected]
    (is (= '[sum-out maximum-out] outputs))
    (is (= '[sum maximum] (mapv first folds)))
    (is (some #{'sum} (tree-seq coll? seq (nth (second folds) 4)))
        "the second certified fold still consumes the first completed fold")
    (is (= '[(clojure.core/float sum) (clojure.core/float maximum)] results))))

(deftest cooperative-maximum-materializes-its-typed-infinity-identity
  (let [operation (certified-maximum-operation)
        scheduled (fold-body/schedule
                   operation {:array-types {'values :float 'out :float}
                              :scalar-types {'rows :long 'width :long}})
        source (:source
                (segop-opencl/generate-segfoldmap-kernel
                 operation :target-dialect :opencl-portable
                 :array-types {'values :float 'out :float}
                 :scalar-types {'rows :long 'width :long}))]
    (is (= :max (get-in scheduled [:body :schedule :reduction-operator])))
    (is (= :max (get-in scheduled [:attributes :reduction-operator])))
    (is (str/includes? source "-INFINITY"))
    (is (str/includes? source "rstr_source_max_f32("))))

(deftest cooperative-minimum-materializes-its-typed-infinity-identity
  (let [operation (certified-minimum-operation)
        scheduled (fold-body/schedule
                   operation {:array-types {'values :float 'out :float}
                              :scalar-types {'rows :long 'width :long}})
        source (:source
                (segop-opencl/generate-segfoldmap-kernel
                 operation :target-dialect :opencl-portable
                 :array-types {'values :float 'out :float}
                 :scalar-types {'rows :long 'width :long}))]
    (is (= :min (get-in scheduled [:body :schedule :reduction-operator])))
    (is (= :min (get-in scheduled [:attributes :reduction-operator])))
    (is (str/includes? source "INFINITY"))
    (is (str/includes? source "rstr_source_min_f32("))))

(deftest cooperative-fold-map-rejects-a-non-tree-workgroup
  (is (= :workgroup-size
         (decline-rule
          #(fold-body/lower
            (certified-operation)
            {:workgroup-size 192
             :array-types {'values :float 'out :float}
             :scalar-types {'rows :long 'width :long}})))))

(deftest portable-schedule-is-a-capped-grid-stride-loop-over-three-ordered-loops
  (let [operation (scheduled-operation)
        {:keys [kernel-body]} (fold-body/lower
                               operation {:array-types {'values :float 'out :float}
                                          :scalar-types {'nsegments :int 'width :int}})
        segment-loop (first (:operations kernel-body))
        first-segment (-> kernel-body :indices last :expression)
        segment-axis-computes (take (count (segop/seg-space-segment-dims (:space operation)))
                                    (:operations segment-loop))
        source-workgroup (get-in operation [:grid :block-size])
        source-cap (first (rest (get-in operation [:grid :num-blocks])))
        loops (filterv #(and (instance? raster.compiler.ir.kernel_body.ForLoop %)
                             (not= :segment-grid-stride (get-in % [:attributes :role])))
                       (operation-tree (:operations kernel-body)))]
    (is (= :segment-grid-stride (get-in segment-loop [:attributes :role])))
    (is (every? #(instance? raster.compiler.ir.kernel_body.IndexCompute %)
                segment-axis-computes)
        "grid-stride segment axes are lexically recomputed as typed index SSA")
    (is (= (mapv :name (segop/seg-space-segment-dims (:space operation)))
           (mapv :id segment-axis-computes)))
    (is (instance? raster.compiler.ir.kernel_body.IndexExpr (:step segment-loop)))
    (is (= :add (:op first-segment)))
    (is (= :mul (:op (first (:arguments first-segment)))))
    (is (every? #(and (instance? raster.compiler.ir.kernel_body.IndexCast %)
                      (= :long (:dtype %))
                      (= :exact (:overflow %)))
                (concat (:arguments (first (:arguments first-segment)))
                        [(second (:arguments first-segment))]))
        "group, workgroup, and lane indices widen before address arithmetic")
    (is (= [source-workgroup] (get-in kernel-body [:launch :workgroup-size])))
    (is (= [(launch/minimum
             (body/index-cast source-cap :long :exact)
             (launch/ceil-div
              (body/index-cast (launch/runtime-value 'nsegments) :long :exact)
              (body/index-cast source-workgroup :long :exact)))]
           (get-in kernel-body [:launch :group-count])))
    (is (zero? (get-in kernel-body [:launch :shared-memory-bytes])))
    (is (= 3 (count loops)) "maximum, denominator, then the final dense map")
    (is (= [:ordered :ordered :ordered]
           (mapv #(get-in % [:attributes :association]) loops)))
    (is (str/includes? (pr-str (second loops)) "maximum")
        "the second fold consumes the completed first fold")
    (is (empty? (:iter-args (last loops))))
    (is (= :final-map (get-in (last loops) [:attributes :role])))
    (is (= ['values] (mapv :buffer (:stable-reads kernel-body))))))

(deftest ordered-fold-map-has-one-complete-scheduled-body-certificate
  (let [operation (scheduled-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :int 'width :int}}
        lowered (fold-body/lower operation options)
        scheduled (fold-body/schedule operation options)
        artifact (segop-opencl/generate-segfoldmap-kernel
                  operation :target-dialect :opencl-portable
                  :array-types (:array-types options) :scalar-types (:scalar-types options))]
    (is (scheduled-body/scheduled-kernel-body? scheduled))
    (is (identical? operation (:source scheduled)))
    (is (= (vec (concat (:arrays lowered) (:outputs lowered) (:scalars lowered)
                        [(:segment-count lowered)]))
           (:arguments scheduled))
        "the private _nseg parameter binds to the derived segment count")
    (is (= (scheduled-body/derive-scalar-bindings (:body scheduled) (:arguments scheduled))
           (:scalar-bindings scheduled)))
    (is (every? #(= :identity (:conversion %)) (:scalar-bindings scheduled)))
    (is (= {:kind :segmented-fold-map
            :uses [{:value 'values :access :read} {:value 'out :access :write}]
            :association :ordered}
           (:effects scheduled)))
    (is (= {:kind :segfoldmap-body-lowering
            :launch-rank 1
            :segment-parallelism :grid-stride-independent
            :association :ordered
            :source-grid (:grid operation)
            :aliasing :no-write-alias}
           (:legality scheduled)))
    (is (= {:mode :exact :policy :declaration-order :reassociation :none}
           (:numerics scheduled)))
    (is (= :scheduled-kernel-body (get-in artifact [:provenance :lowering])))
    (let [certificate (get-in artifact [:provenance :scheduled-operation])]
      (is (identical? operation (:source certificate)))
      (is (= '[values out nsegments width _nseg] (mapv :name (:abi artifact))))
      (is (= [:input :output :scalar :scalar :scalar] (mapv :kind (:abi artifact))))
      (is (= [:float :float :int :int :long] (mapv :dtype (:abi artifact))))
      (is (= (:arguments certificate) (:arguments artifact)))
      (is (= (scheduled-body/realized-launch certificate) (:launch artifact)))
      (is (= artifact
             (scheduled-body/validate-artifact-projection! certificate artifact))))))

(deftest one-certified-schedule-emits-for-all-c-family-targets
  (let [operation (scheduled-operation)]
    (doseq [[target expected]
            [[:opencl-portable :opencl-c]
             [:cuda :cuda-c]
             [:hip :hip-cpp]]]
      (testing (name target)
        (let [kernel (segop-opencl/generate-segfoldmap-kernel
                      operation :target-dialect target
                      :array-types {'values :float 'out :float}
                      :scalar-types {'nsegments :int 'width :int})]
          (is (artifact/kernel-artifact? kernel))
          (is (= expected (:target kernel)))
          (is (= :scheduled-kernel-body (get-in kernel [:provenance :lowering])))
          (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
          (is (= kernel
                 (scheduled-body/validate-artifact-projection!
                  (get-in kernel [:provenance :scheduled-operation]) kernel)))
          (is (= :no-write-alias (get-in kernel [:abi 0 :aliasing])))
          (is (str/includes? (:source kernel) "foldmap_segment +="))
          (is (if (contains? #{:cuda :hip} target)
                (str/includes? (:source kernel) "gridDim.x")
                (str/includes? (:source kernel) "get_num_groups(0)")))
          (is (every? #(<= (int %) 127) (:source kernel))))))))

(deftest graph-emission-closes-the-fold-map-certificate-over-node-scalars-and-effects
  (let [operation (scheduled-operation)
        scalar-types {'nsegments :int 'width :int}
        graph (kgraph/from-segops
               [operation]
               {:inputs #{'values} :outputs #{'out} :dtype :float
                :buffer-specs {'values {:dtype :float
                                        :elements (launch/product 'nsegments 'width)}
                               'out {:dtype :float
                                     :elements (launch/product 'nsegments 'width)}}
                :scalars [(kgraph/scalar 'nsegments :int)
                          (kgraph/scalar 'width :int)]})
        emitted (segop-opencl/generate-kernel-graph
                 graph :target-dialect :opencl-portable
                 :array-types {'values :float 'out :float}
                 :scalar-types scalar-types)
        artifact (get-in emitted [:nodes 0 :operation])
        certificate (get-in artifact [:provenance :scheduled-operation])]
    (is (identical? operation (:source certificate)))
    (is (= certificate (get-in artifact [:attributes :scheduled-kernel-body])))
    (is (= certificate
           (fold-body/validate-against-node! certificate (first (:nodes graph)) graph)))
    (is (= #{'nsegments 'width} (get-in graph [:nodes 0 :scalar-uses])))
    (is (= artifact
           (scheduled-body/validate-artifact-projection! certificate artifact)))))

(deftest fold-map-graph-emission-never-uses-the-segred-validator
  (let [operation (scheduled-operation)
        elements (launch/product 'nsegments 'width)
        graph (kgraph/from-segops
               [operation]
               {:inputs #{'values} :outputs #{'out} :dtype :float
                :buffer-specs {'values {:dtype :float :elements elements}
                               'out {:dtype :float :elements elements}}
                :scalars [(kgraph/scalar 'nsegments :int)
                          (kgraph/scalar 'width :int)]})]
    (with-redefs [segred-body/validate-against-node!
                  (fn [& _]
                    (throw (ex-info "SegRed validator used for SegFoldMap" {})))]
      (is (artifact/kernel-artifact?
           (get-in (segop-opencl/generate-kernel-graph
                    graph :target-dialect :opencl-portable
                    :array-types {'values :float 'out :float}
                    :scalar-types {'nsegments :int 'width :int})
                   [:nodes 0 :operation]))))))

(deftest malformed-fold-map-contracts-decline-before-a-refinement-witness
  (let [operation (scheduled-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :int 'width :int}}
        result (first (:map-results operation))
        cases
        [[:aliasing-contract (assoc operation :aliasing :may-alias)]
         [:storage-contract (update operation :inputs conj 'out)]
         [:result-contract (assoc operation :outputs [] :dtypes [] :map-results [])]
         [:result-contract (assoc operation :dtypes [])]
         [:result-contract (assoc operation :map-results [])]
         [:mapped-extent (assoc operation :extent 'other-width)]
         [:fold-association (assoc-in operation [:folds 0 :association] :associative)]
         [:distinct-outputs (assoc operation
                                   :outputs ['out 'out]
                                   :dtypes [:float :float]
                                   :map-results [result result])]]]
    (doseq [[rule malformed] cases]
      (testing (name rule)
        (is (= rule (decline-rule #(fold-body/schedule malformed options))))))))

(deftest source-kernel-grid-is-an-exact-capped-zero-shared-memory-contract
  (let [operation (scheduled-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :int 'width :int}}
        source-workgroup (get-in operation [:grid :block-size])]
    (is (= :source-grid
           (decline-rule #(fold-body/schedule
                           operation (assoc options :workgroup-size (inc source-workgroup))))))
    (is (= :source-grid-count
           (decline-rule #(fold-body/schedule
                           (assoc-in operation [:grid :num-blocks] 1) options))))
    (is (= :source-grid-shared-memory
           (decline-rule #(fold-body/schedule
                           (assoc-in operation [:grid :shared-mem-bytes] 4) options))))
    (let [large-cap-operation
          (assoc-in operation [:grid :num-blocks]
                    (list 'min Integer/MAX_VALUE
                          (nth (get-in operation [:grid :num-blocks]) 2)))
          first-segment (-> (fold-body/schedule large-cap-operation options)
                            :body :indices last :expression)
          multiplied (first (:arguments first-segment))]
      (is (= :mul (:op multiplied)))
      (is (every? #(= :long (:dtype %)) (:arguments multiplied))
          "even an adversarial positive source cap widens grid operands before multiplication"))))

(deftest graph-certification-closes-fold-map-pointer-dtypes-extents-and-schedule
  (let [operation (scheduled-operation)
        scalar-types {'nsegments :int 'width :int}
        elements (launch/product 'nsegments 'width)
        make-graph
        (fn [input-dtype input-elements]
          (kgraph/from-segops
           [operation]
           {:inputs #{'values} :outputs #{'out} :dtype :float
            :buffer-specs {'values {:dtype input-dtype :elements input-elements}
                           'out {:dtype :float :elements elements}}
            :scalars [(kgraph/scalar 'nsegments :int)
                      (kgraph/scalar 'width :int)]}))
        graph (make-graph :float elements)
        scheduled (fold-body/schedule
                   operation {:array-types {'values :float 'out :float}
                              :scalar-types scalar-types})
        node (first (:nodes graph))
        first-parameter (first (get-in scheduled [:body :parameters]))
        forged-shape
        (-> scheduled
            (assoc-in [:body :parameters 0 :shape] [1])
            (assoc-in [:body :parameters 0 :layout]
                      (layout/row-major [1] (:dtype first-parameter))))
        forged-launch
        (assoc-in scheduled [:body :launch :group-count] [1])
        forged-workgroup
        (assoc-in scheduled [:body :launch :workgroup-size] [1])
        forged-shared
        (assoc-in scheduled [:body :launch :shared-memory-bytes] 4)
        forged-schedule
        (assoc-in scheduled [:body :schedule :workgroup-size] 1)]
    (is (= scheduled (fold-body/validate-against-node! scheduled node graph)))
    (is (= :segfoldmap-storage-extent
           (reason-of #(fold-body/validate-against-node! forged-shape node graph))))
    (let [wrong-extent (make-graph :float (launch/sum elements 1))]
      (is (= :segfoldmap-storage-extent
             (reason-of #(fold-body/validate-against-node!
                          scheduled (first (:nodes wrong-extent)) wrong-extent)))))
    (let [forged-attribute-graph
          (kgraph/from-segops
           [operation]
           {:inputs #{'values} :outputs #{'out} :dtype :float
            :buffer-specs {'values {:dtype :float :elements 1}
                           'out {:dtype :float :elements 1}}
            :scalars [(kgraph/scalar 'nsegments :int)
                      (kgraph/scalar 'width :int)]
            :attributes {:derived-storage-scalars {'nsegments 1 'width 1}}})]
      (is (= :segfoldmap-storage-extent
             (reason-of #(fold-body/validate-against-node!
                          scheduled (first (:nodes forged-attribute-graph))
                          forged-attribute-graph)))
          "descriptive graph attributes cannot forge the storage proof"))
    (let [wrong-dtype (make-graph :double elements)]
      (is (= :scheduled-kernel-body-node-storage-dtype
             (reason-of #(fold-body/validate-against-node!
                          scheduled (first (:nodes wrong-dtype)) wrong-dtype)))))
    (is (= :segfoldmap-schedule-source
           (reason-of #(fold-body/validate-against-node! forged-launch node graph))))
    (doseq [forged [forged-workgroup forged-shared forged-schedule]]
      (is (= :segfoldmap-schedule-source
           (reason-of #(fold-body/validate-against-node! forged node graph)))))))

(deftest graph-storage-replays-the-closed-host-scalar-extent-prefix
  (let [typed (:program (route/attempt source :float {'values :float 'out :float}
                                      {:scalar-types {'nsegments :int 'width :int}}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'values :float 'out :float}
                                 :scalar-types {'nsegments :int 'width :int}}))
        numerical (some #(when (seq (:operations %)) %) (:equations scheduled))
        extent 'derived-extent
        extent-expression (with-meta '(clojure.core/* segments elements)
                            {:raster.type/tag 'long})
        values (-> (:values scheduled)
                   (assoc extent (av/tensor {:dtype :long :shape []}))
                   (update 'values assoc :shape [(list 'value extent)])
                   ;; Opaque graph extent identities are resolver-owned leaves, not nested
                   ;; graph-scalar expressions.  Keep one alongside the expanded input extent.
                   (update 'out assoc :shape [(list 'extent 'input)]))
        scalar-algorithm
        (dialect/make
         (dialect/default-program-facts
          {:values values :inputs '[nsegments width]
           :equations {extent (dialect/default-equation-facts)}})
         [(list '= extent [extent]
                (list 'scalar {:dtypes [:long]} '[nsegments width]
                      (dialect/lambda-form '[segments elements]
                                           [extent-expression])))]
         [extent])
        scalar-equation
        (parallel-program/->ProgramEquation
         extent [:test :derived-extent] nil '[nsegments width] [extent]
         scalar-algorithm [] #{} {:source :test} {:host-only true})
        equations [scalar-equation numerical]
        closed (assoc scheduled
                      :values values
                      :inputs (parallel-program/infer-inputs equations)
                      :equations equations
                      :outputs (:results numerical)
                      :effects (reduce set/union #{} (map :effects equations)))
        graph (equation-graph/make (:algorithm numerical) closed)
        values-buffer (some #(when (= 'values (:id %)) %) (:inputs graph))
        graph-scalar-references
        (reduce set/union #{}
                (map #(into #{}
                             (filter (fn [value] (or (symbol? value) (keyword? value))))
                             (launch/expression-references (:elements %)))
                     (concat (:inputs graph) (:outputs graph) (:temporaries graph))))]
    (is values-buffer)
    (is (= #{'nsegments 'width}
           (launch/expression-references (:elements values-buffer)))
        "the graph allocation replays the preceding typed scalar definition")
    (is (not-any? #{extent}
                  (launch/expression-references (:elements values-buffer))))
    (is (= #{'nsegments 'width}
           (set (map :id (:scalars graph)))))
    (is (= (list 'extent 'input)
           (:elements (some #(when (= 'out (:id %)) %) (:outputs graph))))
        "a compound resolver identity remains opaque without inventing a GraphScalar")
    (is (set/subset? graph-scalar-references (set (map :id (:scalars graph))))
        "every graph-storage extent closes over an explicit integral graph scalar")
    (is (nil? (get-in graph [:attributes :derived-storage-scalars]))
        "the derivation remains a retained program proof, never a graph assertion")
    (let [proof-values (update values 'out assoc :shape [(list 'value extent)])
          proof-closed (assoc closed :values proof-values)
          proof-graph (equation-graph/make (:algorithm numerical) proof-closed)
          operation (first (:operations numerical))
          certificate (fold-body/schedule
                       operation {:array-types {'values :float 'out :float}
                                  :scalar-types {'nsegments :int 'width :int}})
          emitted (segop-opencl/generate-kernel-graph
                   proof-graph
                   :target-dialect :opencl-portable
                   :array-types {'values :float 'out :float}
                   :scalar-types {'nsegments :int 'width :int}
                   :scheduled-equation-algorithm (:algorithm numerical)
                   :scheduled-equation-body proof-closed)]
      (is (= certificate
             (fold-body/validate-against-node!
              certificate (first (:nodes proof-graph)) proof-graph
              (:algorithm numerical) proof-closed)))
      (is (artifact/kernel-artifact? (get-in emitted [:nodes 0 :operation]))
          "graph emission threads the exact retained equation proof into FoldMap"))))

(deftest host-prefix-does-not-hide-later-numerical-inputs
  (let [two-map-source
        '(let* [y (raster.par/pmap i n float
                                   (clojure.core/* 2.0 (clojure.core/aget x i)))
                z (raster.par/pmap j n float
                                   (clojure.core/+ (clojure.core/aget y j)
                                                   (clojure.core/aget b j)))
                _effect (clojure.core/prn y)]
               z)
        options {:dtype :float :target-device :ocl:0
                 :array-types {'x :float 'b :float 'acc :float}
                 :scalar-types {'n :long}}
        typed (:program (route/attempt two-map-source :float (:array-types options)
                                       {:scalar-types (:scalar-types options)}))
        scheduled (:form (segop-lower/segop-lower-pass typed options))
        numerical-equations (:equations scheduled)
        host-id 'graph-proof-host
        values (assoc (:values scheduled) host-id (av/tensor {:dtype :long :shape []}))
        host-algorithm
        (dialect/make
         (dialect/default-program-facts
          {:values values :inputs []
           :equations {host-id (dialect/default-equation-facts)}})
         [(list '= host-id [host-id]
                (list 'scalar {:dtypes [:long]} []
                      (dialect/lambda-form [] [1])))]
         [host-id])
        host-equation
        (parallel-program/->ProgramEquation
         host-id [:test :graph-proof-host] nil [] [host-id]
         host-algorithm [] #{} {:source :test} {:host-only true})
        algorithms (mapv :algorithm numerical-equations)
        numerical-inputs (parallel-program/infer-inputs numerical-equations)
        combined-algorithm
        (dialect/make
         (dialect/default-program-facts
          {:values values
           :inputs numerical-inputs
           :equations (apply merge (map (comp :equations dialect/facts) algorithms))
           :effects (reduce set/union #{} (map (comp :effects dialect/facts) algorithms))})
         (vec (mapcat dialect/equations algorithms))
         (:outputs scheduled))
        equations (into [host-equation] numerical-equations)
        closed (assoc scheduled
                      :values values
                      :inputs (parallel-program/infer-inputs equations)
                      :equations equations)
        graph (equation-graph/make combined-algorithm closed)]
    (is (= 2 (count numerical-equations)))
    (is (some #{'b} numerical-inputs)
        "the second equation introduces an external input absent from the first")
    (is (= 2 (count (:nodes graph))))
    (is (some #{'b} (map :id (:inputs graph))))))

(deftest scalar-captures-retain-their-typed-abi
  (let [scaled-source
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[sum 0.0 :float width
                   (clojure.core/+ sum
                                   (clojure.core/aget values
                                                      (clojure.core/+ (clojure.core/* segment width)
                                                                      index)))]]
                 [(clojure.core/* scale sum)])]
               effect)
        program (:program
                 (route/attempt scaled-source :float {'values :float 'out :float}
                                {:scalar-types {'nsegments :int 'width :int 'scale :float}}))
        operation (-> (segop-lower/segop-lower-pass
                       program {:dtype :float :target-device :ocl:0
                                :array-types {'values :float 'out :float}
                                :scalar-types {'nsegments :int 'width :int 'scale :float}})
                      :form :equations first :operations first)
        kernel (segop-opencl/generate-segfoldmap-kernel
                operation :target-dialect :opencl-portable
                :array-types {'values :float 'out :float}
                :scalar-types {'nsegments :int 'width :int 'scale :float})
        scale-parameter (some #(when (= 'scale (:name %)) %) (:abi kernel))]
    (is (= :float (:dtype scale-parameter)))
    (is (str/includes? (:source kernel) "float scale"))))

(deftest runtime-segment-lengths-admit-a-ragged-fold-domain
  (let [ragged-source
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[row rows]] index 1
                 [[sum 0.0 :float (clojure.core/aget lengths row)
                   (clojure.core/+ sum
                                   (clojure.core/aget values
                                      (clojure.core/unchecked-add
                                       (clojure.core/aget offsets row) index)))]]
                 [(clojure.core/float sum)])]
               effect)
        attempted (route/attempt ragged-source :float
                                 {'values :float 'offsets :long 'lengths :long 'out :float}
                                 {:scalar-types {'rows :long}})
        program (:program attempted)]
    (is program (pr-str (dissoc attempted :program)))
    (when program
      (is (set/subset? #{'values 'offsets 'lengths}
                       (set (:inputs program)))
          "a bound read is a real stable tensor dependency, not a scalar")
      (let [operation (-> (segop-lower/segop-lower-pass
                           program {:dtype :float :target-device :ocl:0
                                    :array-types {'values :float 'offsets :long
                                                  'lengths :long 'out :float}
                                    :scalar-types {'rows :long}})
                          :form :equations first :operations first)
            decline (try
                      (segop-opencl/generate-segfoldmap-kernel
                       operation :target-dialect :opencl-portable
                       :array-types {'values :float 'offsets :long
                                     'lengths :long 'out :float}
                       :scalar-types {'rows :long})
                      nil
                      (catch clojure.lang.ExceptionInfo exception
                        (ex-data exception)))]
        (is (= :segfoldmap-kernel-body-declined (:reason decline)))
        (is (= :index-expression (:missing-rule decline))
            "runtime fold bounds remain fail-closed until their range/capacity is certified")))))

(deftest a-fold-cannot-reference-a-future-accumulator
  (let [bad
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[first 0.0 :float width (clojure.core/+ first future)]
                  [future 0.0 :float width (clojure.core/+ future 1.0)]]
                 [first])]
               effect)]
    (try
      (frontend/form->program
       bad {:dtype :float :array-types {'out :float}
            :scalar-types {'nsegments :int 'width :int}})
      (is false "future fold dependencies must fail validation")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-segmented-fold-map-fold-closure
               (:reason (ex-data exception))))))))

(defn- long-ordered-operation
  ([] (long-ordered-operation source))
  ([source]
  (let [types {'nsegments :long 'width :long}
        program (:program (route/attempt source :float {'values :float 'out :float}
                                         {:scalar-types types}))
        scheduled (:form (segop-lower/segop-lower-pass program
                           {:dtype :float :target-device :ocl:0
                            :array-types {'values :float 'out :float}
                            :scalar-types types}))]
    (first (:operations (first (:equations scheduled)))))))

(deftest ordered-long-final-stores-retain-portable-loop-domain-proof
  (let [operation (long-ordered-operation)
        options {:array-types {'values :float 'out :float}
                 :scalar-types {'nsegments :long 'width :long}}
        checked (:kernel-body (fold-body/lower operation options))
        conditions (body/required-preconditions checked)
        outer (first (:operations checked))
        inner (last (butlast (:operations outer)))
        artifact (segop-opencl/generate-segfoldmap-kernel operation
                   :target-dialect :opencl-portable
                   :array-types (:array-types options) :scalar-types (:scalar-types options))]
    (is (= :segment-grid-stride (get-in outer [:attributes :role])))
    (is (= :final-map (get-in inner [:attributes :role])))
    (is (= checked (body/validate! checked)))
    (is (re-find #"if \(\(ulong\)\([^\n]+<= \(ulong\)\([^\n]+\)\) break;" (:source artifact))
        "dynamic Long loop compares unsigned remaining distance before its increment")
    (is (= conditions (:preconditions artifact)))
    (doseq [[n f] [[0 Long/MIN_VALUE] [0 Long/MAX_VALUE] [Long/MAX_VALUE 0]
                   [Long/MAX_VALUE 1] [1 -1] [2 3]]]
      (is (precondition/check! conditions {'_nseg n 'nsegments (max 1 n) 'width f}) (str [n f])))
    (is (thrown? ArithmeticException
                 (precondition/check! conditions {'_nseg 2 'nsegments 2 'width Long/MAX_VALUE})))
    (doseq [[n d f] [[0 Long/MIN_VALUE Long/MIN_VALUE] [0 0 3]
                     [2 1 3] [2 7 3] [7 2 3]]]
      (is (precondition/check! conditions {'_nseg n 'nsegments d 'width f})
          (str "distinct outer bound and actual divisor " [n d f])))
    (doseq [d [0 -1 Long/MIN_VALUE]
            f [0 -1]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"kernel scalar precondition failed"
                           (precondition/check! conditions {'_nseg 2 'nsegments d 'width f}))
          "axis modulo executes on outer entry before an empty inner loop"))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                         #"resolved launch dimension must be an integer"
                         (precondition/check! conditions {'_nseg 2 'width 3}))
        "the actual modulo divisor binding is required independently of the outer bound")
    (doseq [condition conditions]
      (is (= :kernel-body-index-precondition
             (try (body/validate! (update-in checked [:attributes :required-preconditions]
                                   #(vec (remove #{condition} %)))) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          (pr-str condition)))))

(deftest ordered-affine-facts-do-not-escape-the-inner-loop-or-borrow-another-abi
  (let [operation (long-ordered-operation)
        checked (:kernel-body (fold-body/lower operation
                   {:array-types {'values :float 'out :float}
                    :scalar-types {'nsegments :long 'width :long}}))
        outer (first (:operations checked))
        inner-position (- (count (:operations outer)) 2)
        inner (nth (:operations outer) inner-position)
        store-position (first (keep-indexed #(when (instance? raster.compiler.ir.kernel_body.ScalarStore %2) %1)
                                            (:operations inner)))
        store (nth (:operations inner) store-position)
        coordinate (first (:coordinates store))
        prefix (body/->IndexCompute 'preinner-address-product (first (:arguments coordinate)))
        preinner (-> checked
                     (assoc-in [:operations 0 :operations]
                       (vec (concat (take inner-position (:operations outer)) [prefix]
                                    (drop inner-position (:operations outer)))))
                     (assoc-in [:operations 0 :operations (inc inner-position) :operations store-position :coordinates]
                       [(body/expression :add 'preinner-address-product (:id (:index inner)))]))
        other-abi (-> checked
                      (update :parameters conj (body/->KernelParameter 'other-width :scalar :long [] nil nil :parameter))
                      (assoc-in [:operations 0 :operations inner-position :operations store-position :coordinates]
                        [(body/expression :add (body/expression :mul (:id (:index outer)) 'other-width)
                                              (:id (:index inner)))]))]
    (let [changed-divisor (-> checked
                              (update :parameters conj
                                (body/->KernelParameter 'other-nsegments :scalar :long [] nil nil :parameter))
                              (update :operations
                                #(walk/postwalk-replace {'nsegments 'other-nsegments} %)))
          positive-condition (second (body/quotient-domain-preconditions '_nseg 'nsegments))
          missing-positive (update-in checked [:attributes :required-preconditions]
                             #(vec (remove #{positive-condition} %)))]
      (is (some #{positive-condition} (body/required-preconditions checked)))
      (is (some #(and (instance? raster.compiler.ir.kernel_body.IndexCompute %)
                       (= :mod (get-in % [:expression :op]))
                       (some #{'other-nsegments}
                         (tree-seq coll? seq (:expression %))))
                (tree-seq coll? seq (:operations changed-divisor)))
          "changed divisor survives in the actual modulo definition")
      (doseq [candidate [changed-divisor missing-positive]]
        (is (= :kernel-body-index-precondition
               (try (body/validate! candidate) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "actual modulo divisor and its positivity obligation cannot be borrowed")))
    (is (= :mul (:op (:expression prefix))))
    (is (= :final-map (get-in preinner [:operations 0 :operations (inc inner-position) :attributes :role])))
    (doseq [candidate [preinner other-abi]]
      (is (= :kernel-body-index-precondition
             (try (body/validate! candidate) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "valid typed SSA cannot acquire an inner-loop domain before entry or from equal ABI values"))))

(deftest static-positive-final-map-extent-keeps-its-existing-portable-route
  (let [operation (long-ordered-operation (walk/postwalk-replace {'width 4} source))
        checked (:kernel-body (fold-body/lower operation
                    {:array-types {'values :float 'out :float}
                     :scalar-types {'nsegments :long 'width :long}}))
        conditions (body/required-preconditions checked)
        artifact (segop-opencl/generate-segfoldmap-kernel operation
                   :target-dialect :opencl-portable
                   :array-types {'values :float 'out :float}
                   :scalar-types {'nsegments :long 'width :long})]
    (is (= checked (body/validate! checked)))
    (is (string? (:source artifact)))
    (is (precondition/check! conditions {'_nseg 0 'nsegments 1 'width 4}))
    (is (precondition/check! conditions {'_nseg 2 'nsegments 2 'width 4}))
    (is (thrown? ArithmeticException
                 (precondition/check! conditions {'_nseg Long/MAX_VALUE 'nsegments Long/MAX_VALUE 'width 4})))))

(deftest cooperative-address-domains-come-from-the-real-uniform-guard
  (let [operation (certified-operation)
        options {:array-types {'values :float 'out :float} :scalar-types {'rows :long 'width :long}}
        checked (:kernel-body (fold-body/lower operation options))
        guard (first (:operations checked))
        axis (first (:operations guard))
        scheduled (fold-body/schedule operation options)
        artifact (segop-opencl/generate-segfoldmap-kernel operation
                   :target-dialect :opencl-portable :array-types (:array-types options)
                   :scalar-types (:scalar-types options))
        conditions (body/required-preconditions checked)
        args (mapv #(if (= :scalar (:kind %))
                      {:type (:dtype %) :value (if (= 'width (:name %)) 3 2)}
                      (Object.)) (:abi artifact))
        no-guard (assoc checked :operations (:operations guard))
        wrong-bound (assoc-in checked [:masks 0 :predicates 0 :arguments 1] 'width)
        varying-guard (assoc-in checked [:masks 0 :predicates 0 :arguments 0]
                               (body/index-cast 'foldmap-lane :long :exact))
        premature-axis (-> checked
                           (update :indices conj axis)
                           (assoc-in [:operations 0 :operations] (vec (rest (:operations guard)))))]
    (is (instance? raster.compiler.ir.kernel_body.Guard guard))
    (is (instance? raster.compiler.ir.kernel_body.IndexCompute axis))
    (is (= :mod (get-in axis [:expression :op])))
    (is (= ['foldmap-segment '_nseg] (get-in checked [:masks 0 :predicates 0 :arguments])))
    (is (= checked (body/validate! checked)))
    (is (= conditions (:preconditions scheduled) (:preconditions artifact)))
    (is (= [7] (get-in (kcall/make artifact args {:group-count [7]}) [:geometry :group-count]))
        "oversized scheduling overrides are valid; complete workgroups past the bound are masked")
    (is (precondition/check! conditions {'_nseg 0 'rows Long/MIN_VALUE 'width Long/MIN_VALUE}))
    (is (precondition/check! conditions {'_nseg Long/MAX_VALUE 'rows 1 'width 0}))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"kernel scalar precondition failed"
                         (precondition/check! conditions {'_nseg 2 'rows 0 'width 0})))
    (doseq [candidate [no-guard wrong-bound premature-axis]]
      (is (= :kernel-body-index-precondition
             (try (body/validate! candidate) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "neither a later guard nor a different bound can certify actual modulo/address prefixes"))
    (is (= :kernel-body-index-precondition (reason-of #(body/validate! varying-guard)))
        "a lane predicate cannot supply the segment's missing actual address domain")
    (let [barrier (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                         (body/full-participation))
          uniform-barrier (assoc-in checked [:operations 0 :operations] [barrier])
          varying-barrier (assoc-in varying-guard [:operations 0 :operations] [barrier])
          loop-position (first (keep-indexed
                                 (fn [position operation]
                                   (when (and (instance? raster.compiler.ir.kernel_body.ForLoop operation)
                                              (= :cooperative-lane-fold (get-in operation [:attributes :role])))
                                     position)) (:operations guard)))
          zero-stride (assoc-in checked [:operations 0 :operations loop-position :step] 0)
          escaped (update checked :operations conj
                    (body/->ScalarLoad (body/value 'postguard-load :float) 'values
                      [(body/expression :mul 'foldmap-segment 'width)] nil nil :cached))]
      (is (= uniform-barrier (body/validate! uniform-barrier)))
      (is (= :kernel-body-divergent-workgroup-barrier
             (reason-of #(body/validate! varying-barrier)))
          "valid typed lane-dependent control cannot contain a full-participation barrier")
      (is (instance? raster.compiler.ir.kernel_body.ForLoop
                     (get-in zero-stride [:operations 0 :operations loop-position])))
      (is (= 0 (get-in zero-stride [:operations 0 :operations loop-position :step])))
      (let [error (try (body/validate! zero-stride) nil
                       (catch clojure.lang.ExceptionInfo error error))
            actual-loop (get-in zero-stride [:operations 0 :operations loop-position])
            dynamic-zero (assoc-in checked [:operations 0 :operations loop-position :step]
                           (body/index-cast 0 :long :exact))]
        (is (instance? clojure.lang.ExceptionInfo error))
        (is (= "kernel for-loop requires typed carried values and explicit bounds" (ex-message error)))
        (is (= actual-loop (:loop (ex-data error))))
        (is (= :kernel-body-loop-step-positive (reason-of #(body/validate! dynamic-zero)))
            "valid typed dynamic zero reaches independent step positivity validation"))
      (is (= :kernel-body-index-precondition (reason-of #(body/validate! escaped)))
          "segment guard facts cannot escape into a sibling pointer consumer")
      (is (= :scheduled-kernel-body-scalar-bindings
             (reason-of #(scheduled-body/validate!
                           (update scheduled :scalar-bindings
                             (fn [bindings] (filterv (fn [binding] (not= '_nseg (:parameter binding))) bindings)))))))
      (is (= :scheduled-kernel-body-required-preconditions
             (reason-of #(scheduled-body/validate! (assoc scheduled :preconditions []))))))))

(deftest range-only-loop-intervals-never-become-partial-entered-domains
  (let [upper (body/expression :min 'a 'b)
        info (body/entered-for-loop-info (body/index-cast 0 :long :exact)
               {:type :long :range {:lower Long/MIN_VALUE :upper Long/MAX_VALUE}} 1 :exclusive)
        builder (scalar-expression/make-lowerer
                  {:array-types {} :scalar-types {'a :long 'b :long 'f :long}
                   :arrays #{} :index-scope #{'a 'b 'f 'outer 'inner} :lower-index identity
                   :affine-domain {:index 'outer :bound 'N} :id-prefix "range-only"
                   :decline! (fn [& args] (throw (ex-info "unexpected decline" {:arguments args})))})
        lowered ((:with-loop-index builder) 'inner (body/index-cast 0 :long :exact) upper :exclusive
                  #((:compute builder) :* :long ['inner 'f] {:overflow :trap}))
        checked (body/make
                  {:id :range-only-entered-test
                   :parameters (mapv #(body/->KernelParameter % :scalar :long [] nil nil :parameter) '[N a b f])
                   :indices [(body/->IndexBinding 'group :group 0)
                             (body/->IndexCompute 'outer (body/index-cast 'group :long :exact))]
                   :masks [(body/->Mask :active [(body/predicate :lt 'outer 'N)])]
                   :operations [(body/->Guard :active
                                  [(body/->ForLoop (body/value 'inner :long) (body/index-cast 0 :long :exact)
                                     upper 1 [] (conj (vec (:operations lowered)) (body/->Yield [])) [] {})])]
                   :launch (launch/spec {:workgroup-size [1] :group-count [1] :shared-memory-bytes 0})
                   :attributes {:required-preconditions [{:expression 'N :op :>= :value 0}]}})]
    (is (= {:lower 0 :upper (dec (bigint Long/MAX_VALUE))} (:range info)))
    (is (not (contains? info :entered-loop-index)))
    (is (= :trap (get-in lowered [:operations 0 :expression :options :overflow])))
    (is (= checked (body/validate! checked)))))

(deftest actual-legacy-affine-obligations-keep-their-original-selection
  (let [legacy (body/affine-product-preconditions 'N 'F)
        spec {:id :legacy-affine-selection
              :parameters [(body/->KernelParameter 'values :input :float [1] :global
                             (layout/row-major [1] :float) :operand)
                           (body/->KernelParameter 'N :scalar :long [] nil nil :bound)
                           (body/->KernelParameter 'F :scalar :long [] nil nil :parameter)]
              :stable-reads [(body/stable-read 'values)]
              :indices [(body/->IndexBinding 'group :group 0)
                        (body/->IndexCompute 'outer (body/index-cast 'group :long :exact))]
              :masks [(body/->Mask :active [(body/predicate :lt 'outer 'N)])]
              :operations [(body/->Guard :active
                             [(body/->ForLoop (body/value 'inner :long) (body/index-cast 0 :long :exact)
                                'F 1 [] [(body/->ScalarLoad (body/value 'loaded :float) 'values
                                          [(body/expression :mul 'outer 'F)] nil nil :cached)
                                         (body/->Yield [])] [] {})])]
              :launch (launch/spec {:workgroup-size [1] :group-count [1] :shared-memory-bytes 0})
              :attributes {:required-preconditions legacy}}
        checked (body/make spec)
        activated (body/make (assoc-in spec [:attributes :required-preconditions]
                              (body/entered-affine-address-preconditions 'N 'F)))]
    (is (= checked (body/validate! checked)))
    (is (= legacy (body/required-preconditions checked)))
    (is (= activated (body/validate! activated)))
    (doseq [condition legacy]
      (is (= :kernel-body-index-precondition
             (reason-of #(body/validate!
                           (update-in checked [:attributes :required-preconditions]
                             (fn [conditions] (filterv (fn [c] (not= condition c)) conditions))))))
          "partial legacy requirements cannot silently replace complete entered obligations"))))

(deftest loop-header-obligations-are-collected-before-the-inner-binder
  (doseq [bound-position [:lower :upper]]
    (let [parameters (mapv #(body/->KernelParameter % :scalar :long [] nil nil :parameter) '[N d])
          indices [(body/->IndexBinding 'header-group :group 0)
                   (body/->IndexCompute 'header-flat (body/index-cast 'header-group :long :exact))]
          axis (body/->IndexCompute 'header-axis (body/expression :mod 'header-flat 'd))
          expression (body/expression :add 'header-axis (body/index-cast 1 :long :exact))
          zero (body/index-cast 0 :long :exact)
          lower (if (= bound-position :lower) expression zero)
          upper (if (= bound-position :upper) expression zero)
          loop (body/->ForLoop (body/value 'header-inner :long) lower upper 1 [] [(body/->Yield [])] [] {})
          guard (body/->Guard :header-active [axis loop])
          masks [(body/->Mask :header-active [(body/predicate :lt 'header-flat 'N)])]
          launch-spec (launch/spec {:workgroup-size [1] :group-count [1]})
          base (body/quotient-domain-preconditions 'N 'd)
          builder (scalar-expression/make-lowerer
                    {:array-types {} :scalar-types {'N :long 'd :long} :arrays #{}
                     :index-scope #{'N 'd 'header-flat 'header-axis 'header-inner} :lower-index identity
                     :decline! (fn [& args] (throw (ex-info "unexpected header decline" {:arguments args})))})
          region {:parameters parameters :indices indices :masks masks :launch launch-spec
                  :guard guard :index-definitions [axis] :required-preconditions base}
          _ ((:with-index-region builder) region
              #((:with-loop-index builder) 'header-inner lower upper :exclusive (fn [] :entered)))
          conditions ((:required-preconditions builder))
          header-conditions (vec (remove (set base) conditions))
          checked (body/make {:id [:header-prefix bound-position] :parameters parameters :indices indices
                              :masks masks :launch launch-spec :operations [guard]
                              :attributes {:required-preconditions conditions}})]
      (is (seq header-conditions))
      (is (not-any? #{'header-inner} (mapcat #(concat (launch/expression-references (:expression %))
                                                    (launch/expression-references (:value %))) conditions)))
      (is (= checked (body/validate! checked)))
      (is (precondition/check! conditions {'N 0 'd Long/MIN_VALUE}))
      (is (thrown? clojure.lang.ExceptionInfo (precondition/check! conditions {'N 1 'd 0})))
      (doseq [condition header-conditions]
        (is (= :kernel-body-index-precondition
               (try (body/validate! (assoc-in checked [:attributes :required-preconditions]
                                              (vec (remove #{condition} conditions)))) :accepted
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "an empty inner loop does not erase a bound that was already evaluated")))))
