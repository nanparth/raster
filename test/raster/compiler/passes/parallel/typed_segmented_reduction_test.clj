(ns raster.compiler.passes.parallel.typed-segmented-reduction-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-pass]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]))

(defn- tensor [dtype shape]
  (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}}))

(defn- program []
  (let [body '(+ acc
                 (* (aget lhs (+ (* i inner) l))
                    (aget rhs (+ (* l width) j))))
        algebra (scan/certify-reassociation
                 {:acc 'acc :init 0.0 :lambda body} :float)
        equation
        (list '= 'contract-equation ['C]
              (list 'segmented-reduce
                    {:segment-axes '[[i m] [j n]]
                     :index 'l :extent 'k
                     :accumulators ['acc] :identities [0.0]
                     :dtypes [:float] :algebra [algebra]
                     :attributes {:stable-array-captures ['A 'B]}}
                    [] ['A 'B 'n 'k]
                    (dialect/lambda-form ['acc 'lhs 'rhs 'width 'inner] [body])))
        facts
        (assoc (dialect/default-program-facts
                {:front-end :typed-segmented-reduction-test})
               :values {'A (tensor :float '[m k])
                        'B (tensor :float '[k n])
                        'm (tensor :long [])
                        'n (tensor :long [])
                        'k (tensor :long [])
                        'C (tensor :float '[m n])}
               :inputs '[A B m n k]
               :equations {'contract-equation (dialect/default-equation-facts
                                                {:source :test})})]
    (dialect/make facts [equation] '[C])))

(deftest segmented-semantic-projection-is-device-free-and-exact
  (let [source (program)
        operation (first (lower/lower-typed-segmented-reduce source :ze:0 :dtype :float))
        forbidden (fn [& _] (throw (ex-info "semantic projection discovered target hardware" {})))]
    (with-redefs [hardware/descriptor-for forbidden]
      (let [projected (:operation (lower/project-typed-segmented-reduce
                                  source :dtype :float
                                  :flat-idx (get-in operation [:space :flat-idx])))]
        (is (= operation projected))
        (is (= projected (:operation (lower/project-typed-segmented-reduce
                                      source :dtype :float
                                      :flat-idx (get-in operation [:space :flat-idx])))))
        (is (= (:reduction operation) (:reduction projected)))))))

(deftest semantic-projection-retains-core-shadowing-shape-operands
  (let [source (dialect/remap-values (program) {'m 'seq 'n 'count 'k 'first})]
    (binding [util/*shadowing-locals* #{}]
      (let [projected (:operation (lower/project-typed-segmented-reduce source))]
        (is (= '#{seq count first} (:scalars projected)))
        (is (= '[[i seq] [j count] [l first]]
               (mapv (juxt :name :bound) (get-in projected [:space :dims]))))
        (binding [util/*shadowing-locals* '#{seq count first}]
          (is (= projected (:operation (lower/project-typed-segmented-reduce
                                        source :flat-idx (get-in projected [:space :flat-idx]))))))))))

(deftest segmented-reduction-is-a-typed-functional-equation
  (let [program (program)
        equation (first (dialect/equations program))
        attributes (:attributes (dialect/operation-parts equation))]
    (is (= 'segmented-reduce (dialect/operation-kind equation)))
    (is (= '[[i m] [j n]] (:segment-axes attributes)))
    (is (= '[m n k] (dialect/operation-extents equation)))
    (is (= '[A B n k] (dialect/operation-inputs equation)))
    (is (= '[C] (dialect/outputs program)))))

(deftest generic-fusion-keeps-an-unmatched-segmented-reduction-intact
  (let [source (program)
        [result stats] (fusion/fusion-fixpoint source)]
    (is (= source result))
    (is (= :segmented-reduce
           (:kind (fusion/equation-info (first (dialect/equations result))))))
    (is (= 0 (:vertical stats)))
    (is (= 0 (:horizontal stats)))))

(deftest segmented-reduction-lowers-directly-to-canonical-segred
  (with-redefs [legacy/->SoacContract
                (fn [& _]
                  (throw (ex-info "legacy SoacContract was constructed" {})))]
    (let [operation (first (lower/lower-typed-segmented-reduce
                            (program) :ze:0 :dtype :float))
          dims (:dims (:space operation))
          product (:reduction operation)
          step (first (:results (reduction/fold-region product)))]
      (is (instance? raster.compiler.ir.segop.SegRed operation))
      (is (= '[[i m] [j n] [l k]]
             (mapv (juxt :name :bound) dims)))
      (is (= '#{A B} (:inputs operation)))
      (is (= '#{m n k} (:scalars operation)))
      (is (= '#{C} (:outputs operation)))
      (is (= :segmented (:phase operation)))
      (is (nil? (:schedule operation)))
      (is (reduction/product-reduction? product))
      (is (= :float (first (reduction/dtypes product))))
      (is (= 'C (first (reduction/results product))))
      (is (re-find #"A" (pr-str step)))
      (is (re-find #"B" (pr-str step)))
      (is (not (re-find #"lhs|rhs|width|inner" (pr-str step)))))))

(deftest value-remapping-includes-every-segment-extent
  (let [remapped (dialect/remap-values
                  (program) {'m [:binding 'm] 'n [:binding 'n] 'C [:binding 'C]})
        equation (first (dialect/equations remapped))]
    (is (= [['i [:binding 'm]] ['j [:binding 'n]]]
           (get-in (dialect/operation-parts equation) [:attributes :segment-axes])))
    (is (= [[:binding 'm] [:binding 'n] 'k]
           (dialect/operation-extents equation)))
    (is (= [[:binding 'C]] (dialect/outputs remapped)))))

(deftest typed-parallel-program-schedules-the-equation-without-compatibility-relowering
  (let [algorithm (program)
        values (:values (dialect/facts algorithm))
        source '(raster.par/contract C [[i m] [j n]] [[l k]] body)
        equation
        (parallel-program/->ProgramEquation
         'contract-equation [:binding 'C] source '[A B m n k] '[C]
         algorithm [(first (dialect/equations algorithm))] #{}
         {:source-dialect :typed-soac} {})
        envelope
        (parallel-program/make
         {:dialect :typed-soac :source (list 'let* ['C source] 'C)
          :values values :inputs '[A B m n k] :equations [equation] :outputs '[C]
          :effects #{} :operation? (constantly true)
          :algorithm? (fn [_ candidate] (= candidate (dialect/validate! candidate)))})
        lowered (segop-pass/segop-lower-pass
                 envelope {:target-device :ze:0 :dtype :float})
        scheduled-equation (first (:equations (:form lowered)))
        operation (first (:operations scheduled-equation))]
    (is (dialects/valid-typed-soac-program? envelope)
        "the pipeline boundary admits the TypedSOAC segmented-reduce operation")
    (is (= :segop (:dialect (:form lowered))))
    (is (= 1 (get-in lowered [:stats :typed-soac-reused])))
    (is (= 1 (get-in lowered [:stats :segops-lowered])))
    (is (instance? raster.compiler.ir.segop.SegRed operation))
    (is (= :typed-soac (get-in scheduled-equation [:provenance :source-dialect])))
    (is (empty? (:diagnostics (:form lowered))))))


(deftest rank-zero-contraction-selects-a-legal-scalar-tree-workgroup
  ;; Bounded fixture of the independently observed RTX3070 Laptop limits. Occupancy prefers
  ;; 768 lanes with 48 resident warps; a binary reduction tree must choose a legal power of two.
  (let [descriptor {:backend :cuda :subgroup-size 32 :max-workgroup-size 1024
                    :max-warps-per-sm 48 :max-blocks-per-sm 16 :sm-count 40
                    :shared-memory-per-block 49152}
        source (frontend/form->program
                 (frontend/normalize-source
                   '(let* [result (raster.par/contract out [] [[i n]]
                                    (* (aget x i) (aget y i)) :init (float 0.0))] result))
                 {:dtype :float :array-types {'x :float 'y :float 'out :float}
                  :scalar-types {'n :long}})]
    (is (= 768 (hardware/block-size descriptor 65536 :reduction? true))
        "the frozen target's occupancy choice remains unchanged")
    (doseq [target [descriptor (assoc descriptor :max-warps-per-sm 64)]]
      (let [operations (lower/lower-typed-segmented-reduce source :cuda:0
                                                         :dtype :float :target-descriptor target)]
        (is (seq operations))
        (doseq [operation operations
                :let [size (get-in operation [:grid :block-size])]]
          (is (and (pos? size) (zero? (bit-and size (dec size)))))
          (is (<= size (:max-workgroup-size target)))
          (is (<= (get-in operation [:grid :shared-mem-bytes])
                  (:shared-memory-per-block target)))
          (is (= size (get-in operation [:schedule :workgroup-size]))))))))


(deftest reduction-tree-does-not-invent-capacity-for-an-unsupported-target
  (let [source (frontend/form->program
                 (frontend/normalize-source
                   '(let* [result (raster.par/contract out [] [[i n]]
                                    (* (aget x i) (aget y i)) :init (float 0.0))] result))
                 {:dtype :float :array-types {'x :float 'y :float 'out :float}
                  :scalar-types {'n :long}})
        target {:backend :cuda :subgroup-size 32 :max-workgroup-size 1024
                :max-warps-per-sm 48 :max-blocks-per-sm 16 :shared-memory-per-block 2}]
    (is (= :reduction-tree-target-limits
           (try (lower/lower-typed-segmented-reduce source :cuda:0
                                                   :dtype :float :target-descriptor target)
                nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest scalar-reduction-tree-boundaries-use-only-proved-scratch-capacity
  (doseq [extent [0 1] budget [4 8]]
    (let [source (frontend/form->program
                   (frontend/normalize-source
                     (util/subst-syms
                       {'bound extent}
                       '(let* [result (raster.par/contract out [] [[i bound]]
                                        (* (aget x i) (aget y i)) :init (float 0.0))] result)))
                   {:dtype :float :array-types {'x :float 'y :float 'out :float}})
          target {:backend :cuda :subgroup-size 32 :max-workgroup-size 32
                  :max-warps-per-sm 48 :max-blocks-per-sm 16
                  :shared-memory-per-block budget}
          operations (lower/lower-typed-segmented-reduce source :cuda:0
                                                        :dtype :float :target-descriptor target)]
      (is (= 1 (count operations)))
      (is (= 1 (get-in (first operations) [:grid :block-size])))
      (is (= 4 (get-in (first operations) [:grid :shared-mem-bytes]))))))


(deftest segmented-reduction-read-minima-use-the-shape-rank-convention
  (doseq [bound [0 1 'n]]
    (let [source (frontend/form->program
                   (frontend/normalize-source
                     (util/subst-syms
                       {'bound bound}
                       '(let* [result (raster.par/contract out [] [[i bound]]
                                        (* (aget x i) (aget y i)) :init (float 0.0))] result)))
                   {:dtype :float :array-types {'x :float 'y :float 'out :float}
                    :scalar-types {'n :long}})
          equation (first (dialect/equations source))]
      (is (= {'x bound 'y bound}
             (projection/segmented-reduce-core-read-requirements source equation))
          "a rank-one footprint is its dimension, not a singleton product")))
  (let [source (frontend/form->program
                 '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                (* (clojure.core/aget A (+ (* i k) l))
                                   (clojure.core/aget B (+ (* l n) j))))] step)
                 {:dtype :float :array-types {'A :float 'B :float 'C :float}
                  :scalar-types {'m :long 'n :long 'k :long}})
        equation (first (dialect/equations source))]
    (is (= {'A (launch/product 'm 'k) 'B (launch/product 'k 'n)}
           (projection/segmented-reduce-core-read-requirements source equation))
        "true multi-axis footprints retain the complete product")))

(deftest scalar-two-phase-reduction-recomputes-capped-partial-domain
  (doseq [[bound sample expected-groups] [[1025 1025 3] ['n 8193 8]]]
    (let [source (frontend/form->program
                   (frontend/normalize-source
                     (util/subst-syms
                       {'bound bound}
                       '(let* [result (raster.par/contract out [] [[i bound]]
                                        (* (aget x i) (aget y i)) :init (float 0.0))] result)))
                   {:dtype :float :array-types {'x :float 'y :float 'out :float}
                    :scalar-types {'n :long}})
          target {:backend :cuda :subgroup-size 32 :max-workgroup-size 1024
                  :max-warps-per-sm 48 :max-blocks-per-sm 16 :machine-lanes 1024
                  :shared-memory-per-block 49152}
          [partial terminal :as operations]
          (lower/lower-typed-segmented-reduce source :cuda:0
                                             :dtype :float :target-descriptor target)
          groups (get-in partial [:grid :num-blocks])
          temporary (first (:outputs partial))]
      (is (= 2 (count operations)))
      (is (= 512 (get-in partial [:grid :block-size])))
      (is (= expected-groups (launch/resolve-expression {'n sample} groups)))
      (is (= groups (get-in terminal [:space :dims 0 :bound]))
          "cross-block reads cover exactly the recomputed partial output domain")
      (is (contains? (:inputs terminal) temporary))
      (is (= :block-local (:phase partial)))
      (is (= :cross-block (:phase terminal))))))
