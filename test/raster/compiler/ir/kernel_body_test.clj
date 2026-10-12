(ns raster.compiler.ir.kernel-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]))

(def ^:private matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16})
(def ^:private acc-layout (layout/mma-frag matrix :float))
(def ^:private lhs-layout (layout/dot-operand 0 acc-layout 2 :half))
(def ^:private rhs-layout (layout/dot-operand 1 acc-layout 2 :half))

(deftest concrete-launches-must-fit-the-declared-hardware-index-width
  (doseq [[source dimension accepted rejected]
          [[:group :group-count (inc (long Integer/MAX_VALUE)) (+ 2 (long Integer/MAX_VALUE))]
           [:group-count :group-count (long Integer/MAX_VALUE) (inc (long Integer/MAX_VALUE))]
           [:local :workgroup-size (inc (long Integer/MAX_VALUE)) (+ 2 (long Integer/MAX_VALUE))]
           [:subgroup :workgroup-size (inc (long Integer/MAX_VALUE)) (+ 2 (long Integer/MAX_VALUE))]
           [:lane :workgroup-size (inc (long Integer/MAX_VALUE)) (+ 2 (long Integer/MAX_VALUE))]]]
    (let [index (body/->IndexBinding 'hardware-index source 0)
          kernel (body/make {:id :index-range :indices [index]
                             :launch (launch/spec {:workgroup-size [1] :group-count [1]})})
          geometry #(launch/geometry (assoc {:workgroup-size [1] :group-count [1]}
                                            dimension [%]))]
      (is (= (geometry accepted)
             (body/validate-launch-index-ranges! kernel (geometry accepted))))
      (try
        (body/validate-launch-index-ranges! kernel (geometry rejected))
        (is false (str source " must not overflow int hardware indices"))
        (catch clojure.lang.ExceptionInfo e
          (is (= :kernel-launch-index-range (:reason (ex-data e))))))))
  (let [kernel (body/make {:id :lane-range :indices [(body/->IndexBinding 'lane :lane 0)]
                           :launch (launch/spec {:workgroup-size [1 1 1] :group-count [1 1 1]})})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"hardware index representation"
                         (body/validate-launch-index-ranges!
                          kernel (launch/geometry {:workgroup-size [2147483648 2147483648 4]
                                                   :group-count [1 1 1]})))
        "the proof product must not overflow host arithmetic")))

(defn- minimal-body []
  (body/make
   {:id :matrix
    :parameters [(body/->KernelParameter 'A :input :half [16 16] :global
                                         (layout/row-major [16 16] :half) :lhs)
                 (body/->KernelParameter 'B :input :half [16 16] :global
                                         (layout/row-major [16 16] :half) :rhs)
                 (body/->KernelParameter 'C :output :half [16 16] :global
                                         (layout/row-major [16 16] :half) :result)]
    :indices [(body/->IndexBinding 'group-x :group 0)]
    :masks [(body/->Mask :in-bounds [(body/predicate :lt 'group-x 16)])]
    :fragments [(body/->Fragment :lhs :half [8 16] lhs-layout)
                (body/->Fragment :rhs :half [16 16] rhs-layout)
                (body/->Fragment :acc :float [8 16] acc-layout)]
    :operations [(body/->Guard
                  :in-bounds
                  [(body/->FragmentInit :acc 0.0)
                   (body/->ForLoop
                    (body/value 'k :int) 0 16 16 []
                    [(body/->TileLoad :lhs 'A ['group-x 'k] :in-bounds :cached nil)
                     (body/->TileLoad :rhs 'B ['k 'group-x] :in-bounds :cached nil)
                     (body/->MatrixMad :acc :lhs :rhs matrix)
                     (body/->Yield [])] [] {:unroll true})
                   (body/->TileStore 'C :acc ['group-x 0] :in-bounds nil)])]
    :schedule {:matrix matrix}
    :launch (launch/spec {:workgroup-size [16] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {:kind :matrix-contraction}}))

(deftest tile-input-regions-have-a-closed-typed-element-boundary
  (let [path [:operations 0 :operations 1 :operations 0 :value-region]
        region (body/->ScalarSSARegion
                ['loaded] [] [] :float
                [(body/->ScalarCompute (body/value 'converted :half)
                                       (body/cast-expression 'loaded :half :nearest-even :ieee))]
                'converted :half)
        kernel (-> (minimal-body)
                   (assoc-in [:parameters 0 :dtype] :float)
                   (assoc-in [:parameters 0 :layout] (layout/row-major [16 16] :float))
                   (assoc-in path region))]
    (is (= kernel (body/validate! kernel)))
    (is (thrown? clojure.lang.ExceptionInfo (body/validate! (assoc-in kernel path nil))))
    (doseq [bad [(assoc region :accumulator-dtype :half)
                 (assoc region :result-dtype :float)
                 (assoc region :indices ['group-x])
                 (assoc region :parameters ['loaded 'capture])
                 (assoc region :result 'missing)
                 (assoc-in region [:operations 0 :expression :arguments] ['group-x])
                 (assoc-in region [:operations 0 :expression :options :rounding] nil)
                 (assoc region :operations
                        [(body/->ScalarLoad (body/value 'converted :half) 'B [0 0]
                                            nil nil :cached)])]]
      (is (thrown? clojure.lang.ExceptionInfo (body/validate! (assoc-in kernel path bad)))))))

(deftest a-kernel-body-makes-schedule-decisions-explicit
  (let [kernel (minimal-body)
        guard (first (:operations kernel))
        loop-op (second (:operations guard))]
    (is (body/kernel-body? kernel))
    (is (= :dot-operand (get-in kernel [:fragments 0 :layout :kind])))
    (is (= :mma-frag (get-in kernel [:fragments 2 :layout :kind])))
    (is (= 16 (:step loop-op)))
    (is (= :in-bounds (:mask guard)))
    (is (= matrix (:instruction (nth (:operations loop-op) 2))))))

(deftest kernel-buffer-shapes-retain-symbolic-launch-arithmetic
  (let [extent (launch/ceil-div 'n 256)
        kernel (-> (minimal-body)
                   (assoc-in [:parameters 0 :shape] [extent 16])
                   (assoc-in [:parameters 0 :layout]
                             (layout/row-major [extent 16] :half)))]
    (is (= kernel (body/validate! kernel)))
    (is (= extent (get-in kernel [:parameters 0 :shape 0])))))

(deftest the-verifier-refuses-implicit-or-incompatible-kernel-semantics
  (testing "opaque list arithmetic cannot hide in coordinates"
    (let [kernel (minimal-body)
          bad (assoc-in kernel [:operations 0 :operations 1 :operations 0 :coordinates 0]
                        '(+ group-x 1))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicit index expressions"
                            (body/validate! bad)))))
  (testing "matrix operands must inherit the accumulator distribution"
    (let [kernel (minimal-body)
          wrong-parent (layout/mma-frag matrix :half)
          bad (assoc-in kernel [:fragments 0 :layout]
                        (layout/dot-operand 0 wrong-parent 2 :half))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fragment layouts do not agree"
                            (body/validate! bad)))))
  (testing "all memory guards name declared masks"
    (let [kernel (minimal-body)
          bad (assoc-in kernel [:operations 0 :operations 2 :mask] :missing)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"undeclared mask"
                            (body/validate! bad)))))
  (testing "matrix fragment shapes are fixed by the instruction"
    (let [bad (assoc-in (minimal-body) [:fragments 0 :shape] [8 8])]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fragment shapes"
                            (body/validate! bad)))))
  (testing "tile coordinates match storage rank"
    (let [bad (update-in (minimal-body)
                         [:operations 0 :operations 1 :operations 0 :coordinates]
                         pop)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"buffer rank"
                            (body/validate! bad)))))
  (testing "tile-load dtypes follow storage rather than target inference"
    (let [bad (-> (minimal-body)
                  (assoc-in [:fragments 0 :dtype] :float)
                  (assoc-in [:fragments 0 :layout :dtype] :float))]
      (try
        (body/validate! bad)
        (is false "a tile load accepted a fragment dtype different from its source buffer")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :kernel-body-tile-load-dtype (:reason (ex-data exception))))))))
  (testing "storage shape and dtype agree with their named layout"
    (let [bad-shape (assoc-in (minimal-body) [:parameters 0 :shape] [8 16])
          bad-dtype (assoc-in (minimal-body) [:parameters 0 :dtype] :float)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"shape disagrees"
                            (body/validate! bad-shape)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dtype disagrees"
                            (body/validate! bad-dtype))))))

(deftest buffer-views-retain-parent-ownership-and-explicit-offsets
  (let [kernel (minimal-body)
        kernel (-> kernel
                   (assoc-in [:parameters 0 :shape] [16 16 16])
                   (assoc-in [:parameters 0 :layout]
                             (layout/row-major [16 16 16] :half))
                   (assoc-in [:launch :group-count] [16]))
        view (body/->BufferView 'A-slab 'A
                                (body/leading-slice-offset 'group-x [16 16])
                                [16 16] (layout/row-major [16 16] :half))
        viewed (-> kernel
                   (assoc :views [view])
                   (assoc-in [:operations 0 :operations 1 :operations 0 :buffer] 'A-slab))]
    (is (= viewed (body/validate! viewed)))
    (testing "address factors are exact long leaves, not late or narrowing casts"
      (doseq [offset [(body/expression :mul 'group-x 16 16)
                      (body/index-cast (body/expression :mul 'group-x 16 16) :long :exact)
                      (assoc-in (:element-offset view) [:arguments 0 :overflow] :wrap)
                      (assoc-in (:element-offset view) [:arguments 0 :dtype] :int)]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"widen leaves"
                              (body/validate! (assoc-in viewed [:views 0 :element-offset] offset))))))
    (testing "a view may only reference declared storage"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"undeclared storage"
                            (body/validate! (assoc viewed :views [(assoc view :buffer 'missing)])))))
    (testing "its offset is checked in the scalar/index scope"
      (let [unknown-offset (assoc view :element-offset
                                  (body/expression :mul 'unknown 256))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"outside its scope"
                              (body/validate!
                               (assoc viewed :views [unknown-offset]))))))
    (testing "the parent extent is tied to the hardware axis launch bound"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"launch-bounded"
                            (body/validate! (assoc-in viewed [:launch :group-count 0] 15)))))))

(deftest view-offsets-use-declared-scalar-types
  (let [make-view (fn [extent-type]
                    (body/make
                     {:id :typed-view
                      :parameters [(body/->KernelParameter 'A :input :float [2 'width] :global
                                                            (layout/row-major [2 'width] :float) :input)
                                   (body/->KernelParameter 'width :scalar extent-type [] nil nil :dimension)]
                      :indices [(body/->IndexBinding 'row :group 0)]
                      :views [(body/->BufferView 'slice 'A
                                                (body/leading-slice-offset 'row ['width])
                                                ['width] (layout/row-major ['width] :float))]
                      :launch (launch/spec {:workgroup-size [1] :group-count [2]})}))]
    (doseq [extent-type [:int :long]]
      (is (body/kernel-body? (make-view extent-type))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"only exact widening"
                          (make-view :float)))))

(deftest view-capacity-products-are-checked-before-any-allocation
  (let [resolve-capacity #(launch/resolve-expression {} (apply launch/product %))]
    (is (= 8589934592 (resolve-capacity [2 65536 65536])))
    (is (thrown? ArithmeticException
                 (resolve-capacity [Integer/MAX_VALUE Integer/MAX_VALUE 3 0]))
        "a trailing zero cannot conceal an overflowing address prefix")))

(deftest scalar-regions-are-checked-against-the-ordered-kernel-abi
  (let [semantic-region (body/->ScalarRegion
                ['value 'bias 'scale]
                '(raster.numeric/* (raster.numeric/+ value (aget bias group-x)) scale)
                [{:sym 'bias :map (axis-map/of-axes [['group-x 16]]) :dtype :half}]
                :float)
        region (body/->ScalarSSARegion (:parameters semantic-region) (:operands semantic-region)
                                      ['i 'j] :float [] 'value :float)
        kernel (-> (minimal-body)
                   (update :parameters into
                           [(body/->KernelParameter
                             'bias :input :half [16] :global
                             (layout/row-major [16] :half) :epilogue)
                            (body/->KernelParameter 'scale :scalar :float [] nil nil :epilogue)])
                   (assoc-in [:operations 0 :operations 2 :value-region] region))]
    (is (= kernel (body/validate! kernel)))
    (testing "semantic source regions must be lowered before entering executable KernelBody"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"must be typed scalar SSA"
                            (body/validate! (assoc-in kernel
                                                     [:operations 0 :operations 2 :value-region]
                                                     semantic-region)))))
    (testing "operand identities occupy the ordered prefix after the accumulator"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"ordered prefix"
                            (body/validate!
                             (assoc-in kernel [:operations 0 :operations 2 :value-region
                                               :parameters]
                                       ['value 'scale 'bias])))))
    (testing "every region parameter has exactly one epilogue ABI slot"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"epilogue ABI disagree"
                            (body/validate! (update kernel :parameters pop)))))
    (testing "an operand map and physical ABI shape cannot drift"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"operand and its kernel ABI slot disagree"
                            (body/validate!
                             (assoc-in kernel [:operations 0 :operations 2 :value-region
                                               :operands 0 :map]
                                       (axis-map/of-axes [['group-x 8]]))))))))

(deftest typed-scalar-store-regions-have-closed-ssa-and-a-shared-buffer-abi
  (let [region
        (body/->ScalarSSARegion
         ['value 'A] [{:sym 'A :map (axis-map/of-axes [['i 16] ['j 16]]) :dtype :half}]
         ['i 'j] :half
         [(body/->ScalarLoad (body/value 'loaded :half) 'A ['i 'j] nil nil :cached)
          (body/->ScalarCompute
           (body/value 'sum :half)
           (body/scalar-expression :+ :half ['value 'loaded]))]
         'sum :half)
        kernel (-> (minimal-body)
                   (assoc :stable-reads [(body/stable-read 'A)])
                   (assoc-in [:operations 0 :operations 2 :value-region] region))]
    (is (= kernel (body/validate! kernel)))
    (testing "a shared contraction input does not need a second epilogue ABI slot"
      (is (= ['A 'B 'C] (mapv :id (:parameters kernel)))))
    (testing "uses must follow definitions inside the closed region"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before it is defined"
           (body/validate!
            (assoc-in kernel [:operations 0 :operations 2 :value-region :operations]
                      [(body/->ScalarCompute
                        (body/value 'sum :half)
                        (body/scalar-expression :+ :half ['value 'loaded]))])))))
    (testing "the declared region result dtype is checked against typed SSA"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"wrong dtype"
           (body/validate!
            (assoc-in kernel [:operations 0 :operations 2 :value-region :result-dtype]
                      :float)))))
    (testing "the nested region admits only its small scalar/load vocabulary"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"incomplete"
           (body/validate!
            (update-in kernel [:operations 0 :operations 2 :value-region :operations]
                       conj (body/->ScalarStore 'C ['i 'j] 'sum nil))))))
    (testing "unclaimed epilogue slots cannot drift outside the region boundary"
      (let [bias (body/->KernelParameter 'bias :scalar :float [] nil nil :epilogue)]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"epilogue ABI disagree"
             (body/validate! (update kernel :parameters conj bias))))))))

(defn- scalar-body
  [operations & {:keys [masks stable-reads allocations schedule shared-memory-bytes]
                 :or {masks [] stable-reads [] allocations [] schedule {:subgroup-size 16}
                      shared-memory-bytes 0}}]
  (body/make
   {:id :scalar
    :parameters [(body/->KernelParameter
                  'x :input :float [16] :global
                  (layout/row-major [16] :float) :input)
                 (body/->KernelParameter
                  'y :output :float [1] :global
                  (layout/row-major [1] :float) :result)]
    :indices [(body/->IndexBinding 'group :group 0)
              (body/->IndexBinding 'lane :lane 0)]
    :masks masks
    :stable-reads stable-reads
    :allocations allocations
    :operations operations
    :schedule schedule
    :launch (launch/spec {:workgroup-size [16] :group-count [1]
                          :shared-memory-bytes shared-memory-bytes})
    :provenance {:dialect :test}
    :attributes {:kind :scalar}}))

(defn- dynamic-step-body
  [step]
  (let [kernel (scalar-body [])
        loop (body/->ForLoop
              (body/value 'grid-item :long)
              (body/index-cast 0 :long :exact)
              (body/index-cast 16 :long :exact)
              step [] [(body/->Yield [])] [] {:role :grid-stride})]
    (body/validate!
     (-> kernel
         (update :indices conj (body/->IndexBinding 'group-count :group-count 0))
         (assoc :operations [loop])))))

(deftest dynamic-for-loop-steps-require-an-invariant-positive-proof
  (let [group-count-long (body/index-cast 'group-count :long :exact)
        long-literal #(body/index-cast % :long :exact)
        valid (body/expression :mul group-count-long (long-literal 16))]
    (is (body/kernel-body? (dynamic-step-body valid)))
    (testing "zero and sign-unknown dynamic steps cannot make a progress claim"
      (doseq [step [(body/expression :mul group-count-long (long-literal 0))
                    (body/expression :sub group-count-long group-count-long)]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"provably positive"
                              (dynamic-step-body step)))))
    (testing "literal zero and negative steps remain structurally illegal"
      (doseq [step [0 -1]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicit bounds"
                              (dynamic-step-body step)))))
    (testing "a dynamic step is loop-invariant and uses the induction representation"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"outside its invariant scope"
           (dynamic-step-body (body/index-cast 'missing-step :long :exact))))
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"induction dtype"
           (dynamic-step-body 'group-count))))))

(defn- load-x
  ([] (load-x (body/value 'x-value :float)))
  ([result]
   (body/->ScalarLoad result 'x ['lane] nil nil :cached)))

(defn- reduce-x
  [input]
  (body/->Collective
   (body/value 'sum :float) :reduce :subgroup 16 input :+ nil
   (body/full-participation) :implementation-defined {:overflow :ieee}))

(deftest typed-scalar-memory-control-and-collectives-share-one-body
  (let [predicate (body/->ScalarCompute
                   (body/value 'negative? :predicate)
                   (body/scalar-expression :lt :predicate
                                           ['x-value (body/literal 0.0 :float)]))
        choose (body/->IfRegion
                'negative?
                [(body/->ScalarCompute
                  (body/value 'negated :float)
                  (body/scalar-expression :- :float
                                          [(body/literal 0.0 :float) 'x-value]))
                 (body/->Yield ['negated])]
                [(body/->Yield ['x-value])]
                [(body/value 'magnitude :float)])
        loop-op (body/->ForLoop
                 (body/value 'iteration :int) 0 4 1
                 [(body/->LoopArg (body/value 'acc :float) 'magnitude)]
                 [(body/->ScalarCompute
                   (body/value 'next-acc :float)
                   (body/scalar-expression :+ :float
                                           ['acc (body/literal 1.0 :float)]))
                  (body/->Yield ['next-acc])]
                 [(body/value 'loop-result :float)]
                 {:unroll true})
        kernel (scalar-body
                [(load-x) predicate choose loop-op
                 (reduce-x 'loop-result)
                 (body/->ScalarStore 'y ['group] 'sum nil)])]
    (is (body/kernel-body? kernel))
    (is (= :predicate (get-in kernel [:operations 1 :result :type])))
    (is (= :full (get-in kernel [:operations 4 :participation :kind])))
    (is (launch/launch-spec? (:launch kernel)))))

(deftest elementary-math-realization-is-explicit-not-a-bitwise-attestation
  (let [operators (into #{} (keep (fn [[op descriptor]]
                                  (when (= :target-library (:kernel-body-math-realization descriptor))
                                    op))) intrinsics/table)]
    (is (= #{:sin :cos :tan :exp :log :pow :asin :acos :atan :atan2
             :sinh :cosh :tanh :asinh :acosh :atanh :cbrt :log2 :log10
             :exp2 :expm1 :log1p :hypot} operators))
    (doseq [operator operators dt [:float :double]]
      (let [arguments (vec (repeat (:arity (intrinsics/descriptor operator))
                                   (body/literal 0.5 dt)))
            expression (body/scalar-expression operator dt arguments)
            compute #(scalar-body [(body/->ScalarCompute (body/value 'result dt) %)])]
        (is (= {:math-realization {:kind :target-library :accuracy :implementation-defined}}
               (:options expression)))
        (is (body/kernel-body? (compute expression)))
        (doseq [options [{}
                         {:math-realization {:kind :target-library :accuracy :correctly-rounded}}
                         {:math-realization {:kind :target-library :accuracy :implementation-defined}
                          :overflow :wrap}]]
          (try
            (compute (assoc expression :options options))
            (is false "serialized expressions cannot invent or drop the math realization")
            (catch clojure.lang.ExceptionInfo e
              (is (= :kernel-body-math-realization (:reason (ex-data e)))))))))))

(deftest scalar-ssa-and-conversion-policies-fail-loudly
  (testing "SSA values cannot be used before definition"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"before it is defined"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'result :float)
            (body/scalar-expression :+ :float
                                    ['missing (body/literal 1.0 :float)]))]))))
  (testing "result identities are globally unique"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not globally unique"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'x-value :float)
            (body/scalar-expression :+ :float
                                    ['x-value (body/literal 1.0 :float)]))]))))
  (testing "comparisons produce predicates"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"comparison intrinsic must produce a predicate"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/scalar-expression :lt :float
                                    ['x-value (body/literal 0.0 :float)]))]))))
  (testing "casts cannot leave rounding and overflow behavior implicit"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"rounding and overflow"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/scalar-expression :cast :int ['x-value]))]))))
  (testing "a fully specified cast is legal"
    (is (body/kernel-body?
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'converted :int)
            (body/cast-expression 'x-value :int :toward-zero :saturate))]))))
  (testing "integral arithmetic may carry an explicit wrapping contract"
    (is (body/kernel-body?
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'wrapped :long)
            (body/scalar-expression
             :* :long [(body/literal Long/MAX_VALUE :long)
                       (body/literal 2 :long)]
             {:overflow :wrap}))]))))
  (testing "integral arithmetic cannot omit its overflow semantics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"requires exactly one overflow contract"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'ambiguous :long)
            (body/scalar-expression
             :+ :long [(body/literal 1 :long) (body/literal 2 :long)]))]))))
  (testing "trapping and compiler-proved arithmetic are distinct legal contracts"
    (doseq [policy [:trap :no-overflow]]
      (is (body/kernel-body?
           (scalar-body
            [(body/->ScalarCompute
              (body/value (symbol (name policy)) :long)
              (body/scalar-expression
               :- :long [(body/literal 4 :long) (body/literal 2 :long)]
               {:overflow policy}))])))))
  (testing "wrapping overflow is not a floating-point policy"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"overflow contracts are only defined"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/scalar-expression :+ :float
                                    ['x-value (body/literal 1.0 :float)]
                                    {:overflow :wrap}))]))))
  (testing "wrapping overflow cannot annotate unrelated integral intrinsics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"overflow contracts are only defined"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'bad :int)
            (body/scalar-expression :bit-xor :int
                                    [(body/literal 1 :int) (body/literal 2 :int)]
                                    {:overflow :wrap}))]))))
  (testing "integral sources do not accept a fictitious rounding direction"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'bad :long)
            (body/cast-expression (body/literal 1 :int) :long :up :exact))]))))
  (testing "integral-to-floating conversion states its precision loss"
    (is (body/kernel-body?
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'converted :float)
            (body/cast-expression (body/literal 1 :long)
                                  :float :nearest-even :exact))]))))
  (testing "floating conversions cannot silently request integer wrapping"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/cast-expression 'x-value :int :toward-zero :wrap))]))))
  (testing "FP narrowing states IEEE overflow and nearest-even rounding"
    (is (body/kernel-body?
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'narrowed :half)
            (body/cast-expression 'x-value :half :nearest-even :ieee))]))))
  (testing "IEEE overflow is not an integer conversion policy"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/cast-expression 'x-value :int :toward-zero :ieee))]))))
  (testing "same-width floating casts are exact rather than fictitious narrowing"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/cast-expression 'x-value :float :nearest-even :ieee))]))))
  (testing "floating widening is exact"
    (is (body/kernel-body?
         (scalar-body
          [(body/->ScalarCompute
            (body/value 'widened :float)
            (body/cast-expression (body/literal 1.0 :half) :float :exact :exact))]))))
  (testing "floating narrowing cannot borrow integer saturation semantics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"policies disagree"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :half)
            (body/cast-expression 'x-value :half :nearest-even :saturate))])))))

(deftest scalar-memory-is-ranked-masked-and-typed
  (testing "a load cannot silently reinterpret or widen storage"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"buffer element type"
         (scalar-body [(load-x (body/value 'x-value :int))]))))
  (testing "coordinates are element coordinates with the buffer rank"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"match the buffer rank"
         (scalar-body
          [(body/->ScalarLoad (body/value 'x-value :float) 'x ['group 'lane]
                              nil nil :cached)]))))
  (testing "a false load predicate always has a defined SSA value"
    (let [active (body/->Mask :active [(body/predicate :lt 'lane 8)])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"explicit other value"
           (scalar-body
            [(body/->ScalarLoad (body/value 'x-value :float) 'x ['lane]
                                :active nil :cached)]
            :masks [active])))))
  (testing "an unmasked load cannot retain a meaningless fallback value"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"ignored other"
         (scalar-body
          [(body/->ScalarLoad (body/value 'x-value :float) 'x ['lane]
                              nil (body/literal 0.0 :float) :cached)]))))
  (testing "input-only storage cannot be written"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"writable output storage"
         (scalar-body
          [(load-x) (body/->ScalarStore 'x ['lane] 'x-value nil)])))))

(deftest atomic-updates-require-an-explicit-read-write-contract
  (let [atomic (body/->AtomicRMW 'y [0] (body/literal 1.0 :float) :+ nil)
        inout-body (fn [operation]
                     (-> (scalar-body [])
                         (assoc-in [:parameters 1 :kind] :inout)
                         (assoc :operations [operation])
                         body/validate!))]
    (is (body/kernel-body? (inout-body atomic)))
    (testing "write-only output storage cannot hide a read-modify-write"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"read-write kernel storage"
           (body/validate! (assoc (scalar-body []) :operations [atomic])))))
    (testing "the target-neutral body only accepts a proved update algebra"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"unsupported operator"
           (inout-body (assoc atomic :operator :max)))))
    (testing "the contribution must agree with the resident element dtype"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"contribution type"
           (inout-body (assoc atomic :value (body/literal 1 :int))))))
    (testing "the atomic old value is an optional typed SSA result"
      (is (body/kernel-body?
           (inout-body (assoc atomic :result (body/value 'old :float)))))
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"atomic result type"
           (inout-body (assoc atomic :result (body/value 'old :int))))))))

(deftest structured-regions-prove-yields-and-collective-convergence
  (testing "both if branches terminate with typed yields"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"must terminate in Yield"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'condition :predicate)
            (body/scalar-expression :lt :predicate
                                    ['x-value (body/literal 0.0 :float)]))
           (body/->IfRegion 'condition [] [(body/->Yield [])] [])]))))
  (testing "loop-carried yields match the declared result product"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"yield arity or types"
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'acc :float) 'x-value)]
            [(body/->Yield [(body/literal 1 :int)])]
            [(body/value 'result :float)] {})]))))
  (testing "full-subgroup collectives cannot execute under lane-varying control"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lane-divergent"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'condition :predicate)
            (body/scalar-expression :lt :predicate
                                    ['x-value (body/literal 0.0 :float)]))
           (body/->IfRegion
            'condition
            [(reduce-x 'x-value) (body/->Yield [])]
            [(body/->Yield [])]
            [])]))))
  (testing "collective width agrees with the static launch and schedule"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"subgroup geometry"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'sum :float) :reduce :subgroup 8 'x-value :+ nil
            (body/full-participation) :implementation-defined {:overflow :ieee})]))))
  (testing "collective arithmetic cannot silently change the source dtype semantics"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"arithmetic contract disagrees"
         (scalar-body
          [(load-x)
           (assoc (reduce-x 'x-value) :arithmetic {:overflow :wrap})]))))
  (testing "broadcast source lanes are statically in range"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"statically in range"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'shared :float) :broadcast :subgroup 16 'x-value nil 16
            (body/full-participation) nil nil)]))))
  (testing "broadcasts do not claim a reduction association"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"no reduction association"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'shared :float) :broadcast :subgroup 16 'x-value nil 0
            (body/full-participation) :implementation-defined nil)])))))

(deftest memory-loads-cannot-prove-convergence-without-alias-facts
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"lane-divergent"
       (scalar-body
        [(body/->ScalarLoad (body/value 'shared-load :float) 'x ['group]
                            nil nil :cached)
         (body/->ScalarCompute
          (body/value 'condition :predicate)
          (body/scalar-expression :lt :predicate
                                  ['shared-load (body/literal 0.0 :float)]))
         (body/->IfRegion
          'condition
          [(reduce-x 'shared-load) (body/->Yield [])]
          [(body/->Yield [])]
          [])]))))

(deftest stable-input-loads-prove-only-their-coordinate-uniformity
  (let [stable [(body/stable-read 'x)]
        uniform-branch
        [(body/->ScalarLoad (body/value 'shared-load :float) 'x ['group]
                            nil nil :cached)
         (body/->ScalarCompute
          (body/value 'condition :predicate)
          (body/scalar-expression :lt :predicate
                                  ['shared-load (body/literal 0.0 :float)]))
         (body/->IfRegion
          'condition
          [(reduce-x 'shared-load) (body/->Yield [])]
          [(body/->Yield [])]
          [])]]
    (is (= stable (:stable-reads (scalar-body uniform-branch :stable-reads stable))))
    (testing "a lane-varying address remains lane-varying"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"lane-divergent"
           (scalar-body
            [(load-x)
             (body/->ScalarCompute
              (body/value 'condition :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->IfRegion
              'condition
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])]
            :stable-reads stable))))
    (testing "the contract names unique input parameters"
      (doseq [requirements [[(body/stable-read 'y)]
                            [(body/stable-read 'missing)]
                            [(body/stable-read 'x) (body/stable-read 'x)]]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (scalar-body [] :stable-reads requirements)))))))

(deftest kernel-storage-and-fragments-require-canonical-dtype-facts
  (let [kernel (minimal-body)]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"known dtype"
         (body/validate! (assoc-in kernel [:parameters 0 :dtype] :mystery))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"known dtype"
         (body/validate! (assoc-in kernel [:fragments 0 :dtype] :mystery))))))

(deftest loop-uniformity-is-safe-across-backedges-and-zero-trips
  (testing "a uniform initial carry cannot prove later iterations uniform"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lane-divergent"
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'continue? :predicate)
                             (body/literal true :predicate))]
            [(body/->IfRegion
              'continue?
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])
             (body/->ScalarCompute
              (body/value 'next? :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->Yield ['next?])]
            [(body/value 'finished? :predicate)] {})]))))
  (testing "an explicit inductive invariant admits a genuinely uniform carry"
    (is (body/kernel-body?
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'continue? :predicate)
                             (body/literal true :predicate))]
            [(body/->IfRegion
              'continue?
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])
             (body/->Yield [(body/literal true :predicate)])]
            [(body/value 'finished? :predicate)]
            {:uniform-iter-args #{'continue?}})]))))
  (testing "the verifier checks rather than trusts the claimed backedge invariant"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not preserved by its backedge"
         (scalar-body
          [(load-x)
           (body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'continue? :predicate)
                             (body/literal true :predicate))]
            [(body/->IfRegion
              'continue?
              [(reduce-x 'x-value) (body/->Yield [])]
              [(body/->Yield [])]
              [])
             (body/->ScalarCompute
              (body/value 'next? :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->Yield ['next?])]
            [(body/value 'finished? :predicate)]
            {:uniform-iter-args #{'continue?}})]))))
  (testing "uniformity annotations can name only carried bindings"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"must name carried bindings"
         (scalar-body
          [(body/->ForLoop
            (body/value 'i :int) 0 1 1
            [(body/->LoopArg (body/value 'carry :predicate)
                             (body/literal true :predicate))]
            [(body/->Yield [(body/literal true :predicate)])]
            [(body/value 'result :predicate)]
            {:uniform-iter-args #{'missing}})]))))
  (testing "a zero-trip loop result retains its initial value's variation"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"lane-divergent"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'initial? :predicate)
            (body/scalar-expression :lt :predicate
                                    ['x-value (body/literal 0.0 :float)]))
           (body/->ForLoop
            (body/value 'i :int) 0 0 1
            [(body/->LoopArg (body/value 'carry? :predicate) 'initial?)]
            [(body/->Yield [(body/literal true :predicate)])]
            [(body/value 'result? :predicate)] {})
           (body/->IfRegion
            'result?
            [(reduce-x 'x-value) (body/->Yield [])]
            [(body/->Yield [])]
            [])])))))

(deftest loop-carried-integral-ranges-do-not-become-unchecked-proofs
  (testing "an additive fold cannot certify overflowing prefixes"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not derivable from operand ranges"
         (scalar-body
          [(body/->ForLoop
            (body/value 'i :int) 0 4 1
            [(body/->LoopArg (body/value 'carry :int) (body/literal (dec Integer/MAX_VALUE) :int))]
            [(body/->ScalarCompute
              (body/value 'next-carry :int)
              (body/scalar-expression :+ :int ['carry (body/literal 1 :int)]
                                      {:overflow :no-overflow}))
             (body/->Yield ['next-carry])]
            [(body/value 'loop-result :int)] {})]))))
  (testing "a zero-trip loop cannot borrow its body yield's narrow range"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not derivable from operand ranges"
         (scalar-body
          [(body/->ForLoop
            (body/value 'i :int) 0 0 1
            [(body/->LoopArg (body/value 'carry :int)
                             (body/literal Integer/MIN_VALUE :int))]
            [(body/->Yield [(body/literal 0 :int)])]
            [(body/value 'loop-result :int)] {})
           (body/->ScalarCompute
            (body/value 'unsafe-after-zero-trip :int)
            (body/scalar-expression :- :int ['loop-result (body/literal 1 :int)]
                                    {:overflow :no-overflow}))])))))

(deftest static-additive-loop-ranges-are-independently-derived
  (let [fold (fn [n init term]
               (body/->ForLoop
                (body/value 'i :int) 0 n 1
                [(body/->LoopArg (body/value 'carry :int) (body/literal init :int))]
                [(body/->ScalarCompute
                  (body/value 'next-carry :int)
                  (body/scalar-expression :+ :int ['carry (body/literal term :int)]
                                          {:overflow :no-overflow}))
                 (body/->Yield ['next-carry])]
                [(body/value 'loop-result :int)] {}))]
    (doseq [n [1 3 5]]
      (is (some? (scalar-body [(fold n 7 -2)]))))
    (is (some? (scalar-body [(fold 3 (- Integer/MAX_VALUE 3) 1)])))
    (is (thrown? clojure.lang.ExceptionInfo
                 (scalar-body [(fold 4 (- Integer/MAX_VALUE 3) 1)])))
    (testing "a derived result range is available after the fold"
      (is (some? (scalar-body
                  [(fold 3 0 2)
                   (body/->ScalarCompute
                    (body/value 'after :int)
                    (body/scalar-expression :+ :int ['loop-result (body/literal 1 :int)]
                                            {:overflow :no-overflow}))]))))
    (testing "the term may not depend on the evolving carry"
      (is (thrown? clojure.lang.ExceptionInfo
                   (scalar-body [(assoc-in (fold 3 0 1) [:operations 0 :expression :arguments]
                                           ['carry 'carry])]))))))

(deftest additive-loop-speculation-does-not-reject-ordinary-checked-bodies
  (is (some?
       (scalar-body
        [(body/->ScalarCompute
          (body/value 'external :int)
          (body/scalar-expression :+ :int [(body/literal 0 :int) (body/literal 1 :int)]
                                  {:overflow :no-overflow}))
         (body/->ForLoop
          (body/value 'i :int) 0 3 1
          [(body/->LoopArg (body/value 'carry :int) (body/literal 0 :int))]
          [(body/->ScalarCompute
            (body/value 'term :int)
            (body/scalar-expression :+ :int ['external (body/literal 1 :int)]
                                    {:overflow :no-overflow}))
           (body/->ScalarCompute
            (body/value 'next-carry :int)
            (body/scalar-expression :+ :int ['carry 'term] {:overflow :trap}))
           (body/->Yield ['next-carry])]
          [(body/value 'loop-result :int)] {})]))))

(deftest additive-loop-proof-does-not-borrow-index-arithmetic-ranges
  (is (thrown?
       clojure.lang.ExceptionInfo
       (scalar-body
        [(body/->IndexCompute
          'term (body/expression :min (body/expression :add Integer/MAX_VALUE 1) 1))
         (body/->ForLoop
          (body/value 'i :int) 0 3 1
          [(body/->LoopArg (body/value 'carry :int) (body/literal 0 :int))]
          [(body/->ScalarCompute
            (body/value 'next-carry :int)
            (body/scalar-expression :+ :int ['carry 'term] {:overflow :no-overflow}))
           (body/->Yield ['next-carry])]
          [(body/value 'loop-result :int)] {})]))))

(deftest specification-and-materialized-kernel-share-validation
  (let [kernel (scalar-body [])
        spec (into {} kernel)]
    (is (= spec (with-redefs [body/->KernelBody (fn [& _] (throw (Exception. "materialized")))]
                  (body/validate-spec! spec))))
    (is (= kernel (body/make spec)))
    (doseq [invalid [(assoc spec :operations [:unknown])
                     (assoc spec :parameters nil)]]
      (is (thrown? clojure.lang.ExceptionInfo (body/validate-spec! invalid)))
      (is (thrown? clojure.lang.ExceptionInfo (body/make invalid))))))

(deftest scalar-and-collective-operators-have-semantic-dtype-domains
  (testing "floating intrinsics do not accept integers"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not defined for its operand dtype"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'integer :int)
            (body/cast-expression 'x-value :int :toward-zero :saturate))
           (body/->ScalarCompute
            (body/value 'bad :int)
            (body/scalar-expression :sqrt :int ['integer]))]))))
  (testing "bitwise intrinsics do not accept floats"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not defined for its operand dtype"
         (scalar-body
          [(load-x)
           (body/->ScalarCompute
            (body/value 'bad :float)
            (body/scalar-expression :bit-and :float
                                    ['x-value (body/literal 1.0 :float)]))]))))
  (testing "collective operator legality includes its input dtype"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"not associative for its input dtype"
         (scalar-body
          [(load-x)
           (body/->Collective
            (body/value 'bad :float) :reduce :subgroup 16 'x-value :bit-and nil
            (body/full-participation) :implementation-defined {:overflow :ieee})])))))

(deftest scalar-ssa-cannot-shadow-typed-loop-indices
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"not globally unique"
       (scalar-body
        [(body/->ForLoop
          (body/value 'loop-index :int) 0 1 1 []
          [(body/->ScalarCompute
            (body/value 'loop-index :float)
            (body/scalar-expression :+ :float
                                    [(body/literal 1.0 :float)
                                     (body/literal 2.0 :float)]))
           (body/->Yield [])] [] {})]))))

(deftest hardware-index-axes-must-exist-in-the-launch-contract
  (let [kernel (scalar-body [])]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"invalid hardware source"
         (body/validate! (assoc-in kernel [:indices 0 :axis] 1))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"invalid hardware source"
         (body/validate! (assoc-in kernel [:indices 1 :axis] 1))))))

(defn- scratch-allocation
  ([] (scratch-allocation 'scratch :float [16] 16))
  ([id dtype shape alignment]
   (body/->WorkgroupAllocation id dtype shape (layout/row-major shape dtype) alignment)))

(defn- workgroup-barrier []
  (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                           (body/full-participation)))

(deftest workgroup-storage-has-exact-static-resource-accounting
  (let [allocations [(scratch-allocation 'bytes :byte [3] 1)
                     (scratch-allocation 'values :float [4] 16)]
        plan (body/workgroup-memory-plan allocations)
        kernel (scalar-body
                [(body/->ScalarStore 'values [0] (body/literal 1.0 :float) nil)
                 (workgroup-barrier)
                 (body/->ScalarLoad (body/value 'loaded :float) 'values [0] nil nil :cached)]
                :allocations allocations :shared-memory-bytes 32)]
    (is (= [{:allocation (first allocations) :byte-offset 0 :byte-size 3}
            {:allocation (second allocations) :byte-offset 16 :byte-size 16}]
           (:allocations plan)))
    (is (= 32 (:bytes plan)))
    (is (= 16 (:alignment plan)))
    (is (body/kernel-body? kernel))
    (testing "the launch charge cannot under- or over-state the packed arena"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"shared-memory charge disagree"
           (scalar-body [] :allocations allocations :shared-memory-bytes 31))))
    (testing "dynamic allocation shapes are rejected before target lowering"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"static shape"
           (scalar-body []
                        :allocations [(scratch-allocation 'dynamic :float ['n] 16)]
                        :shared-memory-bytes 0))))
    (testing "the allocation layout cannot drift from its storage contract"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"agree with its layout"
           (scalar-body []
                        :allocations [(assoc (scratch-allocation) :layout
                                             (layout/row-major [8] :float))]
                        :shared-memory-bytes 64))))))

(deftest shared-memory-layouts-have-an-explicit-storage-and-copy-boundary
  (is (body/kernel-body? (fixtures/swizzled-workgroup-memory-body 32)))
  (testing "a shared layout cannot leak onto a global ABI parameter"
    (let [kernel (scalar-body [])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"restricted to workgroup allocations"
           (body/validate!
            (assoc-in kernel [:parameters 0 :layout]
                      (layout/shared-memory [16] :float :identity)))))))
  (testing "a swizzled logical row copy is explicit and cannot promise native overlap"
    (let [kernel (-> (fixtures/async-staging-body 32 :preferred)
                     (assoc-in [:allocations 0 :shape] [32 32])
                     (assoc-in [:allocations 0 :layout]
                               (layout/shared-memory [32 32] :float :xor-32))
                     (assoc-in [:operations 0 :destination-coordinates] [0 0])
                     (assoc-in [:operations 4 :coordinates] [0 'lane])
                     (assoc-in [:launch :shared-memory-bytes] 4096))]
      (is (body/kernel-body? (body/validate! kernel)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"contiguous global-to-workgroup transfer"
                            (body/validate! (assoc-in kernel [:operations 0 :overlap]
                                                      :required)))))))

(deftest workgroup-barriers-require-full-convergent-participation
  (let [barrier (workgroup-barrier)]
    (is (body/kernel-body? (scalar-body [barrier])))
    (testing "a lane-varying branch cannot contain a workgroup barrier"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"divergent control flow"
           (scalar-body
            [(load-x)
             (body/->ScalarCompute
              (body/value 'negative? :predicate)
              (body/scalar-expression :lt :predicate
                                      ['x-value (body/literal 0.0 :float)]))
             (body/->IfRegion 'negative?
                              [barrier (body/->Yield [])]
                              [(body/->Yield [])]
                              [])]))))
    (testing "barrier scope, memory semantics, and participation are explicit"
      (doseq [invalid [(assoc barrier :scope :subgroup)
                       (assoc barrier :memory-spaces #{:global})
                       (assoc barrier :semantics :relaxed)
                       (assoc barrier :participation (body/->Participation :masked))]]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"unsupported synchronization contract"
             (scalar-body [invalid])))))))

(defn- async-stage-copy
  ([] (async-stage-copy [0]))
  ([source-coordinates]
   (body/->AsyncWorkgroupCopy
    'copy-x 'x source-coordinates 'scratch [0] 16 16 :cached :preferred
    (body/full-participation))))

(defn- async-stage-operations
  []
  [(async-stage-copy)
   (body/->AsyncCommit 'copy-group ['copy-x])
   (body/->AsyncWait ['copy-group] 0 :acquire (body/full-participation))
   (workgroup-barrier)
   (body/->ScalarLoad (body/value 'staged :float) 'scratch ['lane] nil nil :cached)])

(deftest async-staging-has-verified-issue-commit-wait-lifetimes
  (let [options [:stable-reads [(body/stable-read 'x)]
                 :allocations [(scratch-allocation)]
                 :shared-memory-bytes 64]]
    (is (body/kernel-body? (apply scalar-body (async-stage-operations) options)))
    (testing "the source must stay stable for the entire asynchronous lifetime"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"stable-read contract"
           (scalar-body (async-stage-operations)
                        :allocations [(scratch-allocation)]
                        :shared-memory-bytes 64))))
    (testing "a commit closes exactly the copies issued since the previous commit"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"close every copy"
           (apply scalar-body
                  (assoc (async-stage-operations) 1
                         (body/->AsyncCommit 'copy-group ['different-copy]))
                  options))))
    (testing "waits consume the oldest group prefix and state remaining depth"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"consume an oldest prefix"
           (apply scalar-body
                  (assoc (async-stage-operations) 2
                         (body/->AsyncWait ['copy-group] 1 :acquire
                                           (body/full-participation)))
                  options))))
    (testing "wait completion does not silently imply cross-thread visibility"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before a workgroup barrier"
           (apply scalar-body
                  (vec (concat (subvec (async-stage-operations) 0 3)
                               (subvec (async-stage-operations) 4)))
                  options))))
    (testing "staged storage cannot be consumed while its copy group is incomplete"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before its async wait"
           (apply scalar-body
                  [(async-stage-copy)
                   (body/->ScalarLoad (body/value 'too-early :float)
                                      'scratch ['lane] nil nil :cached)]
                  options))))
    (testing "the cooperative base address must be workgroup-uniform"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"issued uniformly"
           (apply scalar-body
                  (assoc (async-stage-operations) 0 (async-stage-copy ['lane]))
                  options))))
    (testing "async coordinates may use prior uniform scalar SSA values"
      (is (body/kernel-body?
           (apply scalar-body
                  (into [(body/->ScalarCompute
                          (body/value 'source-base :int)
                          (body/scalar-expression
                           :+ :int ['group (body/literal 0 :int)]
                           {:overflow :no-overflow}))]
                        (assoc (async-stage-operations) 0
                               (async-stage-copy ['source-base])))
                  options))))
    (testing "undefined async coordinate SSA values still fail in the typed verifier"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"before it is defined"
           (apply scalar-body
                  (assoc (async-stage-operations) 0 (async-stage-copy ['missing-base]))
                  options))))))

(deftest pipelined-for-carries-a-verified-rotating-async-queue
  (let [kernel (fixtures/pipelined-staging-body 16 :preferred)
        pipeline-index 4]
    (is (body/kernel-body? kernel))
    (is (= :exact (get-in kernel [:operations pipeline-index :attributes :tail-policy])))
    (is (= ['pipeline-group-a 'pipeline-group-b]
           (get-in kernel [:operations pipeline-index :async-results])))
    (testing "the tail policy is part of the scheduled loop contract"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"explicit tail policy"
           (body/validate! (assoc-in kernel
                                     [:operations pipeline-index :attributes]
                                     {})))))
    (testing "every carried group is the complete ordered queue at loop entry"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"complete ordered async queue"
           (body/validate!
            (-> kernel
                (assoc-in [:operations pipeline-index :async-iter-args]
                          [(body/->AsyncLoopArg 'carry-a 'warm-group-a)])
                (assoc-in [:operations pipeline-index :async-results]
                          ['pipeline-group-a]))))))
    (testing "a stage cannot be refilled before all workgroup readers finish"
      (let [pipeline-operations (get-in kernel [:operations pipeline-index :operations])]
        (is (thrown-with-msg?
             clojure.lang.ExceptionInfo #"destination is still live"
             (body/validate!
              (assoc-in kernel [:operations pipeline-index :operations]
                        (vec (concat (subvec pipeline-operations 0 3)
                                     (subvec pipeline-operations 4)))))))))
    (testing "the backedge preserves each rotating stage position"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"preserve its rotating async stages"
           (body/validate!
            (assoc-in kernel [:operations pipeline-index :operations 4 :elements] 8)))))
    (testing "pending pipeline results must be drained outside the loop"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"lifetime crosses a structured region boundary"
           (body/validate! (assoc kernel :operations
                                  (subvec (:operations kernel) 0 5))))))))

(deftest pipelined-carries-cannot-discard-checked-quotient-ancestry
  (let [original (fixtures/pipelined-staging-body 16 :preferred)
        quotient (body/->ScalarCompute
                  (body/value 'pipeline-quotient :long)
                  (body/scalar-expression :quot :long [(body/literal 2 :long) (body/literal 1 :long)]))
        factor (body/->ScalarCompute
                (body/value 'pipeline-address-factor :long)
                (body/scalar-expression :cast :long [(body/literal Long/MAX_VALUE :long)]
                                       {:rounding :exact :overflow :exact}))
        candidate (fn [initial yielded upper address?]
                    (let [pipeline (-> (get-in original [:operations 4])
                                       (assoc :upper upper
                                              :iter-args [(body/->LoopArg (body/value 'pipeline-scalar-carry :long) initial)]
                                              :results [(body/value 'pipeline-scalar-result :long)])
                                       (update :operations
                                               #(let [operations (assoc-in % [(dec (count %)) :values] [yielded])]
                                                  (if address?
                                                    (vec (concat (pop operations)
                                                                 [(body/->ScalarStore
                                                                   'y [(body/->IndexExpr :mul ['pipeline-scalar-carry 'pipeline-address-factor])]
                                                                   (body/literal 0.0 :float) nil)
                                                                  (peek operations)])) operations))))]
                      (assoc original :operations
                             (vec (concat (subvec (:operations original) 0 4)
                                          [quotient factor pipeline]
                                          (subvec (:operations original) 5))))))
        ordinary (candidate (body/literal 0 :long) (body/literal 0 :long) 2 false)]
    (is (= original (body/validate! original)))
    (is (= ordinary (body/validate! ordinary))
        "the real rotating async queue still accepts ordinary untagged scalar carries")
    (doseq [kernel [(candidate 'pipeline-quotient (body/literal 0 :long) 0 false)
                    (candidate (body/literal 0 :long) 'pipeline-quotient 2 false)]]
      (is (= kernel (body/validate! kernel))
          "harmless checked quotient value-only carries retain zero-trip and backedge semantics"))
    (doseq [kernel [(candidate 'pipeline-quotient (body/literal 0 :long) 2 true)
                    (candidate (body/literal 0 :long) 'pipeline-quotient 2 true)]]
      (is (= :kernel-body-index-precondition
             (try (body/validate! kernel) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "initial or later-backedge quotient ancestry cannot hide a genuinely overflowing address"))))

(defn- guarded-storage-while-kernel [overflow proof]
  (let [cast (fn [id source]
               (body/->ScalarCompute (body/value id :long)
                                     (body/scalar-expression :cast :long [source]
                                                             {:rounding :exact :overflow :exact})))
        load (fn [id offset]
               (body/->ScalarLoad (body/value id :int) 'rp [offset] nil nil :cached))]
    (body/make
     {:id :independent-guarded-while
      :parameters [(body/->KernelParameter 'rp :input :int [2] :global
                                           (layout/row-major [2] :int) :input)]
      :operations [(load 'start32 0) (cast 'start 'start32)
                   (body/->WhileLoop
                    [(body/->LoopArg (body/value 'k :long) 'start)
                     (body/->LoopArg (body/value 'sum :double) (body/literal 0.0 :double))]
                    [(load 'end32 1) (cast 'end 'end32)
                     (body/->ScalarCompute (body/value 'done :predicate)
                                          (body/scalar-expression :ge :predicate ['k 'end]))
                     (body/->IfRegion 'done
                                      [(body/->Yield [(body/literal false :predicate)])]
                                      [(body/->Yield [(body/literal true :predicate)])]
                                      [(body/value 'continue :predicate)])
                     (body/->Yield ['continue])]
                    [(body/->ScalarCompute
                      (body/value 'next-k :long)
                      (body/scalar-expression :+ :long ['k (body/literal 1 :long)]
                                              (cond-> {:overflow overflow} proof (assoc :proof proof))))
                     (body/->ScalarCompute (body/value 'next-sum :double)
                                          (body/scalar-expression :+ :double ['sum (body/literal 1.0 :double)]))
                     (body/->Yield ['next-k 'next-sum])]
                    [(body/value 'result-k :long) (body/value 'result-sum :double)] {:association :ordered})]
      :launch (launch/spec {:workgroup-size [1] :group-count [1]})
      :provenance {:dialect :test} :attributes {:kind :scalar}})))

(deftest guarded-while-invariant-is-reconstructed-independently
  (is (some? (guarded-storage-while-kernel :trap nil))
      "ordinary checked typed IR is valid before portable proof qualification")
  (is (some? (guarded-storage-while-kernel
              :no-overflow {:kind :typed-scalar-range
                            :lower (inc (bigint Integer/MIN_VALUE))
                            :upper (bigint Integer/MAX_VALUE)}))))

(deftest guarded-while-hostile-recurrences-lose-the-range-proof
  (let [proof {:kind :typed-scalar-range :lower (inc (bigint Integer/MIN_VALUE))
               :upper (bigint Integer/MAX_VALUE)}
        ;; Start from the independently validated ordinary loop. Every mutation preserves
        ;; its typed shape; failures must be arithmetic/proof failures, not IR setup errors.
        ordinary (guarded-storage-while-kernel :trap nil)
        claimed (assoc-in ordinary [:operations 2 :operations 0 :expression :options]
                          {:overflow :no-overflow :proof proof})
        reject (fn [kernel]
                 (try (body/validate! kernel) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (some? (body/validate! claimed)))
    (doseq [hostile
            [(assoc-in claimed [:operations 2 :condition-operations 2 :expression :op] :lt)
             (assoc-in claimed [:operations 2 :operations 0 :expression :arguments 1]
                       (body/literal 2 :long))
             (assoc-in claimed [:operations 2 :operations 0 :expression :op] :-)
             (assoc-in claimed [:operations 2 :operations 2 :values 0] 'k)
             (assoc-in claimed [:operations 2 :operations 0 :expression :options :proof :upper] 10)
             ;; This bounded rule does not admit unknown Long endpoints. This is not a
             ;; mathematical claim that dominating k<e cannot prove Long k+1 by another rule.
             ;; A valid Long endpoint parameter has no Int32-storage provenance.
             (-> claimed
                 (update :parameters conj (body/->KernelParameter 'unbounded-end :scalar :long [] nil nil :parameter))
                 (assoc-in [:operations 2 :condition-operations 2 :expression :arguments 1] 'unbounded-end))]]
      (is (= :kernel-body-intrinsic-overflow (reject hostile))))) )

(deftest guarded-while-direct-condition-and-zero-trip-results
  (let [ordinary (guarded-storage-while-kernel :trap nil)
        proof {:kind :typed-scalar-range :lower (inc (bigint Integer/MIN_VALUE))
               :upper (bigint Integer/MAX_VALUE)}
        claimed (assoc-in ordinary [:operations 2 :operations 0 :expression :options]
                          {:overflow :no-overflow :proof proof})
        condition (get-in claimed [:operations 2 :condition-operations])
        direct (assoc-in claimed [:operations 2 :condition-operations]
                         (conj (assoc (subvec condition 0 3) 2
                                      (assoc-in (nth condition 2) [:expression :op] :lt))
                               (body/->Yield ['done])))]
    (is (some? (body/validate! direct)))
    ;; Exit/zero-trip range refinement is intentionally absent: the increment proof must
    ;; not become an unrelated post-loop arithmetic certificate.
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not derivable from operand ranges"
                         (body/validate!
                          (update claimed :operations conj
                                  (body/->ScalarCompute
                                   (body/value 'after :long)
                                   (body/scalar-expression :+ :long ['result-k (body/literal 1 :long)]
                                                           {:overflow :no-overflow}))))))
    (doseq [start [Integer/MIN_VALUE -7 0 Integer/MAX_VALUE]
            end [Integer/MIN_VALUE -2 0 Integer/MAX_VALUE]
            :when (< start end)]
      (let [next-k (inc (bigint start))]
        (is (<= (inc (bigint Integer/MIN_VALUE)) next-k (bigint Integer/MAX_VALUE)))))))

(deftest guarded-while-compared-carry-and-complete-backedge-are-required
  (let [base (guarded-storage-while-kernel :trap nil)
        loop (nth (:operations base) 2)
        other-carry (-> base
                        (assoc-in [:operations 2 :iter-args]
                                  (conj (:iter-args loop)
                                        (body/->LoopArg (body/value 'other-k :long) 'start)))
                        (assoc-in [:operations 2 :results]
                                  (conj (:results loop) (body/value 'other-result :long)))
                        (assoc-in [:operations 2 :operations 2 :values]
                                  ['next-k 'next-sum 'other-k])
                        (assoc-in [:operations 2 :condition-operations 2 :expression :arguments 0]
                                  'other-k))
        alternate (-> base
                      (assoc-in [:operations 2 :operations]
                                (into (subvec (:operations loop) 0 2)
                                      [(body/->ScalarCompute (body/value 'choose-next :predicate)
                                                             (body/scalar-expression :lt :predicate ['k (body/literal 0 :long)]))
                                       (body/->IfRegion 'choose-next
                                                       [(body/->Yield ['next-k])]
                                                       [(body/->Yield ['k])]
                                                       [(body/value 'chosen-k :long)])
                                       (body/->Yield ['chosen-k 'next-sum])])))]
    (doseq [ordinary [other-carry alternate]]
      (is (some? (body/validate! ordinary)) "hostile recurrence itself remains valid checked IR")
      (is (= :kernel-body-intrinsic-overflow
             (try (body/validate!
                   (assoc-in ordinary [:operations 2 :operations 0 :expression :options]
                             {:overflow :no-overflow
                              :proof {:kind :typed-scalar-range :lower (inc (bigint Integer/MIN_VALUE))
                                      :upper (bigint Integer/MAX_VALUE)}}))
                  nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest ceil-div-closes-the-emitted-addition-before-division
  (let [kernel (fn [n d type]
                 (body/make
                  {:id :checked-ceil-prefix
                   :parameters [(body/->KernelParameter 'a :input :long [1] :global
                                  (layout/row-major [1] :long) :operand)]
                   :operations [(body/->ScalarCompute (body/value 'n type)
                                  (body/scalar-expression :cast type [(body/literal n type)]
                                    {:rounding :exact :overflow :exact}))
                                (body/->ScalarCompute (body/value 'd type)
                                  (body/scalar-expression :cast type [(body/literal d type)]
                                    {:rounding :exact :overflow :exact}))
                                (body/->ScalarLoad (body/value 'v :long) 'a
                                  [(body/expression :ceil-div 'n 'd)] nil nil :cached)]
                   :launch (launch/spec {:workgroup-size [1] :group-count [1]})}))
        reason (fn [f] (try (f) :accepted
                           (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (some? (kernel Integer/MAX_VALUE Integer/MAX_VALUE :long)))
    (is (some? (kernel 0 1 :long)))
    (doseq [[n d] [[Long/MAX_VALUE 1] [(dec Long/MAX_VALUE) 2]
                   [1 0] [1 -1] [Long/MIN_VALUE 1]]]
      (is (= :kernel-body-index-precondition (reason #(kernel n d :long)))))
    (is (= :kernel-body-index-precondition (reason #(kernel Integer/MAX_VALUE 1 :int)))
        "Int n+d overflow is not repaired by the representable final quotient")))

(deftest exclusive-entry-retains-a-closed-hardware-lower-sign
  (let [base {:id :actual-lower-sign
              :parameters [(body/->KernelParameter 'a :input :long [1] :global
                             (layout/row-major [1] :long) :operand)]
              :indices [(body/->IndexBinding 'g :group 0)
                        (body/->IndexCompute 'lower (body/index-cast 'g :long :exact))]
              :operations [(body/->ScalarCompute (body/value 'upper :long)
                             (body/scalar-expression :cast :long [(body/literal 16 :long)]
                               {:rounding :exact :overflow :exact}))
                           (body/->ForLoop (body/value 'i :long) 'lower 'upper 1 []
                             [(body/->ScalarLoad (body/value 'v :long) 'a
                                [(body/expression :floor-div 'i (body/index-cast 1 :long :exact))]
                                nil nil :cached) (body/->Yield [])] [] {})]
              :launch (launch/spec {:workgroup-size [1] :group-count [1]})}
        checked (body/make base)
        opaque (-> checked
                   (update :parameters conj (body/->KernelParameter 'opaque :scalar :int [] nil nil :parameter))
                   (assoc-in [:indices 1 :expression] (body/index-cast 'opaque :long :exact)))
        forged (update opaque :parameters
                       #(assoc % (dec (count %)) (assoc (last %) :nonnegative? true :index-origin-safe? true)))
        unsafe (assoc-in checked [:indices 1 :expression]
                         (body/index-cast (body/expression :mul 'g 256) :long :exact))
        negative (assoc-in checked [:indices 1 :expression] (body/index-cast -1 :long :exact))
        inclusive (assoc-in checked [:operations 1 :attributes :upper-bound] :inclusive)
        reason (fn [f] (try (f) :accepted
                           (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= checked (body/validate! checked)))
    (doseq [candidate [opaque forged unsafe negative inclusive]]
      (is (= :kernel-body-index-precondition (reason #(body/validate! candidate)))
          "unknown/signed/overflowing/inclusive bounds cannot borrow the closed exclusive entry"))))


(deftest guarded-quotient-keeps-the-actual-numerator-range-through-remainder
  (let [params [(body/->KernelParameter 'x :input :float [1] :global (layout/row-major [1] :float) :operand)
                (body/->KernelParameter 'N :scalar :long [] nil nil :bound)
                (body/->KernelParameter 'rows :scalar :long [] nil nil :parameter)]
        hardware [(body/->IndexBinding 'g :group 0)]
        launch (launch/spec {:workgroup-size [8] :group-count [1]})
        conditions (vec (distinct (concat (body/quotient-domain-preconditions 'N 8)
                                          (body/quotient-domain-preconditions 'N 'rows))))
        condition (body/->ScalarCompute
                   (body/value 'active :predicate)
                   (body/scalar-expression :lt :predicate
                     [(body/cast-expression 'g :long :exact :exact)
                      (body/cast-expression 'N :long :exact :exact)]))
        defs [(body/->IndexCompute 'q (body/expression :floor-div
                                      (body/index-cast 'g :long :exact)
                                      (body/index-cast 8 :long :exact)))
              (body/->IndexCompute 'r (body/expression :mod 'q 'rows))]
        branch (body/->IfRegion 'active defs [(body/->Yield [])] [])
        region {:parameters params :indices hardware :launch launch :required-preconditions conditions
                :condition-definition condition :if-region branch :index-definitions defs}
        values (:values (body/scheduled-index-region-info region))
        rejection (fn [f]
                    (try (f) {:outcome :accepted}
                         (catch Throwable e
                           {:outcome :rejected :class (.getName (class e))
                            :message (.getMessage e) :data (ex-data e)})))
        assert-address-rejection (fn [f]
                                   (let [outcome (rejection f)]
                                     (is (= :rejected (:outcome outcome)) (pr-str outcome))
                                     (is (= :kernel-body-index-precondition
                                            (get-in outcome [:data :reason])) (pr-str outcome))))
        max-q (quot Integer/MAX_VALUE 8)
        make-kernel (fn [parameters indices requirements]
                      (body/make
                       {:id :bounded-hardware-quotient :parameters parameters :indices indices :launch launch
                        :attributes {:required-preconditions requirements}
                        :operations [condition
                         (assoc branch :then-operations
                           (into defs [(body/->ScalarLoad
                                        (body/value 'loaded :float) 'x
                                        [(body/expression :mul 'r (body/index-cast 64 :long :exact))]
                                        nil nil :cached)
                                       (body/->Yield [])]))]}))]
    (is (= {:lower 0 :upper max-q} (get-in values ['q :range])))
    (is (= {:lower 0 :upper max-q} (get-in values ['r :range])))
    (is (= conditions (get-in values ['r :conditional-integer :requirements])))
    (is (body/kernel-body? (make-kernel params hardware conditions)))
    ;; The early r*64 prefix fits, but multiplication by MAX before a later zero
    ;; does not. Final algebraic zero cannot erase the ordered C intermediate.
    (let [scale (body/->ScalarCompute
                 (body/value 'huge-scale :long)
                 (body/cast-expression (body/literal Long/MAX_VALUE :long) :long :exact :exact))]
      (is (body/kernel-body?
           (body/make {:id :typed-long-scale :parameters params :indices hardware :launch launch
                       :operations [scale]})))
      (assert-address-rejection
       #(body/make
       {:id :late-prefix-overflow :parameters params :indices hardware :launch launch
        :attributes {:required-preconditions conditions}
        :operations [scale condition
                     (assoc branch :then-operations
                       (into defs [(body/->ScalarLoad
                                    (body/value 'loaded :float) 'x
                                    [(body/expression :mul 'r (body/index-cast 64 :long :exact)
                                      'huge-scale
                                      (body/index-cast 0 :long :exact))]
                                    nil nil :cached)
                                   (body/->Yield [])]))]})))
    (is (true? (precondition/check! conditions {'N 0 'rows Long/MIN_VALUE})))
    (doseq [rows [0 -1]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (precondition/check! conditions {'N 1 'rows rows}))))
    (doseq [i (range (count conditions))]
      (let [changed (vec (concat (take i conditions) (drop (inc i) conditions)))]
        (assert-address-rejection #(make-kernel params hardware changed))))
    (let [escape (body/make
                  {:id :hardware-only :parameters params :indices hardware :launch launch})]
      (doseq [operations [(into defs [(body/->ScalarLoad
                                       (body/value 'loaded :float) 'x
                                       [(body/expression :mul 'r (body/index-cast 64 :long :exact))]
                                       nil nil :cached)])
                          [condition (assoc branch
                                       :then-operations [(body/->Yield [])]
                                       :else-operations (into defs
                                          [(body/->ScalarLoad (body/value 'loaded :float) 'x
                                            [(body/expression :mul 'r (body/index-cast 64 :long :exact))]
                                            nil nil :cached)
                                           (body/->Yield [])]))]]]
        (assert-address-rejection
         #(body/validate! (assoc escape :operations operations
                                       :attributes {:required-preconditions conditions})))))
    (let [wide-params (conj params (body/->KernelParameter 'wide :scalar :long [] nil nil :parameter))
          wide-indices [(body/->IndexCompute 'g
                          (body/expression :max 'wide (body/index-cast 0 :long :exact)))]]
      (assert-address-rejection #(make-kernel wide-params wide-indices conditions)))
    (let [abi-params (conj params (body/->KernelParameter 'g :scalar :long [] nil nil :parameter))]
      (assert-address-rejection #(make-kernel abi-params [] conditions)))
    (let [unsafe [(body/->IndexBinding 'raw-group :group 0)
                  (body/->IndexCompute 'g (body/index-cast
                                          (body/expression :mul 'raw-group 8) :long :exact))]]
      (assert-address-rejection #(make-kernel params unsafe conditions)))))


(deftest computed-exclusive-loop-keeps-its-actual-maximum-without-an-abi-alias
  (let [parameters [(body/->KernelParameter 'N :scalar :long [] nil nil :bound)]
        indices [(body/->IndexBinding 'group :group 0)
                 (body/->IndexCompute 'idx (body/index-cast 'group :long :exact))]
        actual-launch (launch/spec {:workgroup-size [8] :group-count [1]})
        condition (body/->ScalarCompute (body/value 'active :predicate)
                     (body/scalar-expression :lt :predicate ['idx 'N]))
        domain [{:expression 'N :op :>= :value 0}]
        branch (body/->IfRegion 'active [(body/->Yield [])] [(body/->Yield [])] [])
        values (:values (body/scheduled-index-region-info
                         {:parameters parameters :indices indices :launch actual-launch
                          :condition-definition condition :if-region branch :index-definitions []
                          :required-preconditions domain}))
        scalar-info (ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!)
        product-expression (body/scalar-expression :* :long ['idx 'N] {:overflow :no-overflow})
        product-transfer (body/conditional-integer-transfer product-expression values)
        product-requirements (:required-preconditions product-transfer)
        values (with-meta values (assoc (meta values) :required-preconditions
                                        (vec (distinct (concat domain product-requirements)))))
        product-info (scalar-info product-expression values)
        values (assoc values 'computed-product product-info)
        upper-expression (body/scalar-expression :+ :long ['computed-product (body/literal 1 :long)]
                                                {:overflow :no-overflow})
        upper-transfer (body/conditional-integer-transfer upper-expression values)
        requirements (vec (distinct (concat domain product-requirements (:required-preconditions upper-transfer))))
        values (with-meta values (assoc (meta values) :required-preconditions requirements))
        upper-info (scalar-info upper-expression values)
        entered (body/entered-for-loop-info (body/index-cast 0 :long :exact) upper-info 1 :exclusive)
        loop (body/->ForLoop (body/value 'entered :long) (body/index-cast 0 :long :exact)
                            'computed-upper 1 [] [(body/->Yield [])] [] {})
        operations [(body/->ScalarCompute (body/value 'computed-product :long) product-expression)
                    (body/->ScalarCompute (body/value 'computed-upper :long) upper-expression) loop
                    (body/->Yield [])]
        kernel (body/make {:id :computed-loop-domain :parameters parameters :indices indices
                           :launch actual-launch :attributes {:required-preconditions requirements}
                           :operations [condition (assoc branch :then-operations operations)]})]
    (is (= 'N (get-in entered [:conditional-integer :bound])))
    (is (nil? (:entered-loop-index entered)) "the computed SSA upper is not an ABI extent")
    (is (= 6 (launch/resolve-expression {'N 3} (get-in entered [:conditional-integer :upper]))))
    (is (true? (precondition/check! requirements {'N 3})))
    (is (= kernel (body/validate! kernel)))
    (is (thrown? ArithmeticException (precondition/check! requirements {'N Long/MAX_VALUE})))
    (is (nil? (body/entered-for-loop-info 0 upper-info 2 :exclusive)))
    (is (nil? (body/entered-for-loop-info 0 upper-info 1 :inclusive)))
    (let [loop-values (assoc values 'entered (assoc-in entered [:conditional-integer :domain-index] 'entered))]
      (is (not-any? #(= 'entered (:index %)) (:entered-loop-domains (meta loop-values))))
      (is (nil? (get-in (body/conditional-index-info (body/expression :mod 'entered 'N) loop-values)
                       [:conditional-integer]))
          "activation N does not prove the computed induction variable is below N"))
    (let [raw-upper (body/expression :add (body/expression :mul 'idx 'N)
                                                (body/index-cast 1 :long :exact))
          raw-info (body/conditional-index-info raw-upper values)
          raw-requirements (vec (distinct (concat domain (:required-preconditions raw-info)
                                                    (body/quotient-domain-preconditions 'N 'N))))
          actual-header (assoc loop :upper raw-upper)
          actual-scope (body/scheduled-index-region-info
                        {:parameters parameters :indices indices :launch actual-launch
                         :condition-definition condition
                         :if-region (assoc branch :then-operations [actual-header (body/->Yield [])])
                         :index-definitions [] :inner-loop actual-header :required-preconditions raw-requirements})
          actual-values (:values actual-scope)]
      (is (= 6 (launch/resolve-expression {'N 3}
                       (get-in actual-values ['entered :conditional-integer :upper]))))
      (is (not-any? #(= 'entered (:index %)) (:entered-loop-domains (meta actual-values)))
          "the actual scoped replay must not publish activation N as the computed binder extent")
      (is (= 2 (quot 6 3)) "N3/idx2 enters six below the actual computed upper seven")
      (is (> (*' (quot 6 3) Long/MAX_VALUE) Long/MAX_VALUE)
          "the falsely inferred quotient zero would conceal a genuinely overflowing product")
      (let [pointer (body/->KernelParameter 'x :input :float [1] :global
                                            (layout/row-major [1] :float) :operand)
            outcome (try
                      (body/make
                       {:id :computed-quotient-correlation :parameters (conj parameters pointer)
                        :indices indices :launch actual-launch
                        :attributes {:required-preconditions raw-requirements}
                        :operations [(body/->ScalarCompute (body/value 'huge-scale :long)
                                                     (body/cast-expression (body/literal Long/MAX_VALUE :long)
                                                                           :long :exact :exact))
                                     condition
                                     (assoc branch :then-operations
                                            [(assoc actual-header :operations
                                                    [(body/->ScalarLoad (body/value 'loaded :float) 'x
                                                       [(body/expression :mul
                                                          (body/expression :floor-div 'entered 'N) 'huge-scale)] nil nil :cached)
                                                     (body/->Yield [])])
                                             (body/->Yield [])])]})
                      {:outcome :accepted}
                      (catch clojure.lang.ExceptionInfo error {:outcome :rejected :data (ex-data error)}))]
        (is (= :rejected (:outcome outcome)) (pr-str outcome))
        (is (= :kernel-body-index-precondition (get-in outcome [:data :reason])) (pr-str outcome))))
    (doseq [requirement product-requirements]
      (let [outcome (try (body/validate!
                         (assoc-in kernel [:attributes :required-preconditions]
                                   (vec (remove #{requirement} requirements))))
                        {:outcome :accepted}
                        (catch clojure.lang.ExceptionInfo error {:outcome :rejected :data (ex-data error)}))]
        (is (= :rejected (:outcome outcome)) (pr-str outcome))))))


(deftest additive-storage-recurrence-validates-addresses-in-the-real-header-scope
  (let [input (fn [type] (body/->KernelParameter 'x :input type [4] :global
                                                (layout/row-major [4] type) :operand))
        loop (fn [type coordinate upper]
               (body/->ForLoop
                (body/value 't :int) 0 upper 1
                [(body/->LoopArg (body/value 'carry :int) (body/literal 0 :int))]
                [(body/->ScalarLoad (body/value 'loaded type) 'x [coordinate] nil nil :cached)
                 (body/->ScalarCompute (body/value 'term :int)
                                      (body/cast-expression 'loaded :int :exact :exact))
                 (body/->ScalarCompute (body/value 'next-carry :int)
                                      (body/scalar-expression :+ :int ['carry 'term] {:overflow :no-overflow}))
                 (body/->Yield ['next-carry])]
                [(body/value 'sum :int)] {}))
        make (fn [type coordinate upper extra]
               (body/make (merge {:id :address-scoped-additive-byte :parameters [(input type)]
                                 :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                                 :operations [(loop type coordinate upper)]} extra)))
        coordinate (body/expression :add 't 1)
        kernel (make :byte coordinate 3 {})]
    (is (= kernel (body/validate! kernel)))
    ;; Three unconstrained byte loads are safe Int terms. Address t+1 needs the
    ;; actual t0..2 header; that bound is never borrowed into the numeric term.
    (is (= 381 (* 3 Byte/MAX_VALUE)))
    (is (= -384 (* 3 Byte/MIN_VALUE)))
    (let [outcome (try (make :int coordinate 3 {}) {:outcome :accepted}
                       (catch clojure.lang.ExceptionInfo error {:outcome :rejected :data (ex-data error)}))]
      (is (= :rejected (:outcome outcome)) (pr-str outcome))
      (is (contains? #{:kernel-body-intrinsic-overflow :kernel-body-intrinsic-overflow-proof}
                     (get-in outcome [:data :reason])) (pr-str outcome))
      (is (> (*' 3 Integer/MAX_VALUE) Integer/MAX_VALUE)
          "loaded Int storage cannot borrow the address induction range as a term bound"))
    (let [unsafe-coordinate (body/expression :add
                              (body/index-cast (body/expression :mul 'group 256) :long :exact)
                              (body/index-cast 't :long :exact))
          outcome (try (make :byte unsafe-coordinate 3
                             {:indices [(body/->IndexBinding 'group :group 0)]
                              :launch (launch/spec {:workgroup-size [8] :group-count [1]})})
                       {:outcome :accepted}
                       (catch clojure.lang.ExceptionInfo error {:outcome :rejected :data (ex-data error)}))]
      (is (= :rejected (:outcome outcome)) (pr-str outcome))
      (is (= :kernel-body-index-precondition (get-in outcome [:data :reason])) (pr-str outcome)))))


(deftest raw-safe-loop-upper-retains-only-the-actual-child-domain
  (let [parameters [(body/->KernelParameter 'N :scalar :long [] nil nil :bound)
                    (body/->KernelParameter 'M :scalar :long [] nil nil :bound)]
        indices [(body/->IndexBinding 'group :group 0)
                 (body/->IndexCompute 'idx (body/index-cast 'group :long :exact))]
        actual-launch (launch/spec {:workgroup-size [8] :group-count [1]})
        conditions (body/quotient-domain-preconditions 'N 8)
        condition (body/->ScalarCompute (body/value 'active :predicate)
                                       (body/scalar-expression :lt :predicate ['idx 'N]))
        definitions [(body/->IndexCompute 'q (body/expression :floor-div 'idx (body/index-cast 8 :long :exact)))]
        upper (body/expression :add 'q (body/index-cast 1 :long :exact))
        header (body/->ForLoop (body/value 'j :long) (body/index-cast 0 :long :exact) upper 1
                              [] [(body/->Yield [])] [] {})
        branch (body/->IfRegion 'active (into definitions [header (body/->Yield [])]) [(body/->Yield [])] [])
        region {:parameters parameters :indices indices :launch actual-launch
                :condition-definition condition :if-region branch :index-definitions definitions
                :inner-loop header :required-preconditions conditions}
        replay (body/scheduled-index-region-info region)
        values (:values replay)
        actual-upper (body/conditional-index-info upper values)]
    (is (= 'N (get-in actual-upper [:conditional-integer :bound])))
    (is (= 'N (get-in values ['j :conditional-integer :bound])))
    (is (nil? (get-in values ['j :entered-loop-index])))
    (is (= 0 (launch/resolve-expression {'N 0} (get-in values ['j :conditional-integer :upper]))))
    (is (= 1 (launch/resolve-expression {'N 9} (get-in values ['j :conditional-integer :upper]))))
    (is (= (quot Integer/MAX_VALUE 8)
           (launch/resolve-expression {'N Long/MAX_VALUE} (get-in values ['j :conditional-integer :upper]))))
    (is (= (set conditions) (set (:requirements replay)))
        "ordinary-safe addition keeps evaluated child conditions without adding a loose host cap")
    (is (not-any? #(= 'j (:index %)) (:entered-loop-domains (meta values))))
    (let [kernel (body/make {:id :raw-safe-loop-upper :parameters parameters :indices indices
                             :launch actual-launch :attributes {:required-preconditions conditions}
                             :operations [condition branch]})]
      (is (= kernel (body/validate! kernel))))
    (let [other-conditions (body/quotient-domain-preconditions 'M 8)
          other-region (-> region
                           (assoc :required-preconditions other-conditions)
                           (assoc-in [:condition-definition :expression :arguments] ['idx 'M]))
          other-values (:values (body/scheduled-index-region-info other-region))
          mixed (body/conditional-index-info (body/expression :add 'q 'other-q)
                                            (assoc values 'other-q (get other-values 'q)))]
      (is (nil? (:conditional-integer mixed))
          "separately reconstructed activation bounds must not be merged by a safe numeric sum"))))

(deftest additive-proof-does-not-borrow-an-external-numeric-index-range
  (let [indices [(body/->IndexBinding 'lane :local 0)
                 (body/->IndexCompute 'outer-term (body/expression :mod 'lane 2))]
        scope (:values (body/initial-scheduled-index-info
                        {:parameters [] :indices indices
                         :launch (launch/spec {:workgroup-size [8] :group-count [1]})}))
        operation (body/->ForLoop (body/value 't :int) 0 3 1
                                  [(body/->LoopArg (body/value 'carry :int) (body/literal 0 :int))]
                                  [(body/->ScalarCompute (body/value 'next :int)
                                     (body/scalar-expression :+ :int ['carry 'outer-term] {:overflow :no-overflow}))
                                   (body/->Yield ['next'])]
                                  [(body/value 'sum :int)] {})
        derive (ns-resolve 'raster.compiler.ir.kernel-body 'additive-loop-ranges!)]
    (is (= {:lower 0 :upper 1} (get-in scope ['outer-term :range])))
    (is (nil? (derive operation (assoc scope 'carry {:type :int :range {:lower 0 :upper 0}})
                     {:claimed (atom #{})})))
    ;; This is an optional recurrence-analysis contract, not a claim that the
    ;; concrete bounded index would overflow. External range provenance remains excluded.
    ))


(deftest compound-abi-divisors-preserve-checked-source-prefixes-and-active-conditions
  (let [wide #(body/index-cast % :long :exact)
        divisor (body/expression :add (wide 'M) (wide 1))
        host-divisor (launch/sum (wide (launch/runtime-value 'M)) (wide 1))
        requirements (body/quotient-domain-preconditions 'N host-divisor)
        indices [(body/->IndexBinding 'compound-group :group 0)
                 (body/->IndexCompute 'compound-flat (wide 'compound-group))]
        masks [(body/->Mask :compound-active [(body/predicate :lt 'compound-flat (wide 'N))])]
        parameters (fn [mtype]
                     [(body/->KernelParameter 'N :scalar :long [] nil nil :bound)
                      (body/->KernelParameter 'M :scalar mtype [] nil nil :parameter)
                      (body/->KernelParameter 'x :input :float [1] :global
                                             (layout/row-major [1] :float) :operand)])
        make (fn [mtype divisor conditions]
               (body/make
                {:id :compound-divisor-address :parameters (parameters mtype) :indices indices :masks masks
                 :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                 :operations [(body/->Guard :compound-active
                                [(body/->IndexCompute 'compound-axis (body/expression :mod 'compound-flat divisor))
                                 (body/->ScalarLoad (body/value 'loaded :float) 'x ['compound-axis] nil nil :cached)])]
                 :attributes {:required-preconditions conditions}}))
        kernel (make :int divisor requirements)]
    ;; x[1] makes this an arithmetic/ancestry test; it is not a memory-capacity proof.
    (is (= kernel (body/validate! kernel)))
    (is (precondition/check! requirements {'N 3 'M 2}))
    (doseq [m [-1 -2]]
      (is (thrown? clojure.lang.ExceptionInfo (precondition/check! requirements {'N 1 'M m})))
      (is (precondition/check! requirements {'N 0 'M m})))
    (doseq [condition requirements]
      (is (= :kernel-body-index-precondition
             (try (make :int divisor (vec (remove #{condition} requirements))) :accepted
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= :kernel-body-index-precondition
           (try (make :long divisor requirements) :accepted
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "full Long M+1 must close its evaluated addition before division")
    (is (= :kernel-body-index-precondition
           (try (make :int (wide (body/expression :add 'M 1)) requirements) :accepted
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "casting after an overflowing Int source prefix cannot certify the divisor")))


(deftest entered-affine-cohorts-preserve-the-actual-enforced-outer-policy
  (let [parameters [(body/->KernelParameter 'N :scalar :long [] nil nil :bound)
                    (body/->KernelParameter 'width :scalar :long [] nil nil :parameter)
                    (body/->KernelParameter 'x :input :float [1] :global
                                           (layout/row-major [1] :float) :operand)]
        zero (body/index-cast 0 :long :exact)
        coordinate (body/expression :add (body/expression :mul 'row 'width) 'j)
        inner (body/->ForLoop (body/value 'j :long) zero 'width 1 []
                             [(body/->ScalarLoad (body/value 'loaded :float) 'x [coordinate] nil nil :cached)
                              (body/->Yield [])] [] {})
        outer (body/->ForLoop (body/value 'row :long) zero 'N 1 []
                             [inner (body/->Yield [])] [] {})
        make #(body/make {:id :entered-affine-owner-cohort :parameters parameters
                          :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                          :operations [outer] :attributes {:required-preconditions %}})
        raw (body/entered-affine-address-preconditions 'N 'width)
        signed (body/entered-affine-address-preconditions (launch/maximum 'N 0) 'width)]
    ;; Both are arithmetic-only witnesses; x[1] does not certify memory capacity.
    (doseq [conditions [raw signed]]
      (let [kernel (make conditions)]
        (is (= kernel (body/validate! kernel)))
        (is (precondition/check! conditions {'N 2 'width 3}))
        (is (thrown? ArithmeticException
              (precondition/check! conditions {'N 3 'width Long/MAX_VALUE})))
        (is (= :kernel-body-index-precondition
               (try (make []) :accepted
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
    (is (= :kernel-body-index-precondition
           (try (make (vec (remove #{{:expression 'N :op :>= :value 0}} raw))) :accepted
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "retained raw capacity caps do not attest the signed cohort after its outer policy is removed")
    (doseq [n [Long/MIN_VALUE -1]]
      (is (thrown? clojure.lang.ExceptionInfo (precondition/check! raw {'N n 'width Long/MIN_VALUE})))
      (is (precondition/check! signed {'N n 'width Long/MIN_VALUE})))
    (is (precondition/check! raw {'N 0 'width Long/MIN_VALUE}))
    (is (precondition/check! signed {'N 0 'width Long/MIN_VALUE}))))


(deftest inclusive-computed-loop-entry-keeps-the-real-upper-and-exit-safety
  (let [parameters [(body/->KernelParameter 'N :scalar :long [] nil nil :bound)
                    (body/->KernelParameter 'x :input :float [1] :global
                                           (layout/row-major [1] :float) :operand)]
        wide #(body/index-cast % :long :exact)
        indices [(body/->IndexBinding 'inclusive-group :group 0)
                 (body/->IndexCompute 'inclusive-flat (wide 'inclusive-group))]
        masks [(body/->Mask :inclusive-active [(body/predicate :lt 'inclusive-flat (wide 'N))])]
        upper (body/->IndexCompute 'inclusive-upper (body/expression :mod 'inclusive-flat (wide 1)))
        conditions (body/quotient-domain-preconditions 'N 1)
        header (body/->ForLoop (body/value 'inclusive-index :long) (wide 0) 'inclusive-upper 1 []
                              [(body/->Yield [])] [] {:upper-bound :inclusive})
        guard (body/->Guard :inclusive-active [upper header])
        region {:parameters parameters :indices indices :masks masks
                :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                :guard guard :inner-loop header :index-definitions [upper]
                :required-preconditions conditions}
        replay (body/scheduled-index-region-info region)
        entered (get-in replay [:values 'inclusive-index])
        kernel (body/make {:id :inclusive-computed-header :parameters parameters :indices indices :masks masks
                           :launch (:launch region) :operations [guard]
                           :attributes {:required-preconditions conditions}})]
    (is (= kernel (body/validate! kernel)))
    (is (= {:lower 0 :upper 0} (:range entered))
        "inclusive upper zero enters the body once, rather than subtracting one")
    (is (= 0 (launch/resolve-expression {'N 1} (get-in entered [:conditional-integer :upper]))))
    (is (= 'N (get-in entered [:conditional-integer :bound])))
    (is (nil? (:entered-loop-index entered)))
    (is (nil? (:abi-scalar-identity entered)))
    (is (not-any? #(= 'inclusive-index (:index %))
                  (:entered-loop-domains (meta (:values replay))))
        "the activation bound is not an index<N quotient identity")
    (let [values (:values replay)
          huge (body/scalar-expression :+ :long ['inclusive-upper (body/literal Long/MAX_VALUE :long)]
                                      {:overflow :no-overflow
                                       :proof {:kind :typed-scalar-range :lower Long/MAX_VALUE :upper Long/MAX_VALUE}})
          actual-upper ((ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!) huge values)]
      (is (= {:lower Long/MAX_VALUE :upper Long/MAX_VALUE} (:range actual-upper)))
      (is (nil? (body/entered-for-loop-info (wide 0) actual-upper 1 :inclusive))
          "a representable upper MAX cannot certify the inclusive exiting increment")
      (let [max-compute (body/->ScalarCompute (body/value 'inclusive-max :long) huge)
            unsafe-loop (assoc header :upper 'inclusive-max
                               :operations [(body/->ScalarLoad (body/value 'past-max :float) 'x
                                              [(body/expression :add 'inclusive-index (wide 1))]
                                              nil nil :cached)
                                            (body/->Yield [])])]
        (is (= :kernel-body-index-precondition
               (try (body/validate! (assoc kernel :operations
                                     [(assoc guard :operations [upper max-compute unsafe-loop])])) :accepted
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "final actual ForLoop cannot borrow a bounded inclusive fact at MAX")))
    (let [negative ((ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!) (body/cast-expression (body/literal -1 :long) :long :exact :exact)
                                     (:values replay))]
      (is (nil? (body/entered-for-loop-info (wide 0) negative 1 :inclusive))))
    (let [outside (:values (body/initial-scheduled-index-info
                            {:parameters parameters :indices (conj indices upper) :launch (:launch region)}))]
      (is (nil? (body/entered-for-loop-info (wide 0)
                   (body/conditional-index-info 'inclusive-upper outside) 1 :inclusive))
          "ordinary remainder range alone does not recreate the lost active conditional domain"))
    (is (body/kernel-body? (body/validate! (assoc-in kernel [:attributes :required-preconditions] [])))
        "modulo one remains independently bounded even without activation obligations")
    (let [dynamic-parameters (conj parameters (body/->KernelParameter 'D :scalar :long [] nil nil :parameter))
          dynamic-upper (assoc upper :expression (body/expression :mod 'inclusive-flat 'D))
          dynamic-conditions (body/quotient-domain-preconditions 'N 'D)
          dynamic-header (assoc header :operations
                                [(body/->ScalarLoad (body/value 'dynamic-loaded :float) 'x
                                                    [(body/expression :add 'inclusive-index (wide 1))]
                                                    nil nil :cached)
                                 (body/->Yield [])])
          dynamic-guard (assoc guard :operations [dynamic-upper dynamic-header])
          dynamic-kernel (body/make (assoc kernel :parameters dynamic-parameters :operations [dynamic-guard]
                                          :attributes {:required-preconditions dynamic-conditions}))]
      (is (= dynamic-kernel (body/validate! dynamic-kernel)))
      (let [dynamic-replay (body/scheduled-index-region-info
                            (assoc region :parameters dynamic-parameters
                                   :guard dynamic-guard :inner-loop dynamic-header
                                   :index-definitions [dynamic-upper]
                                   :required-preconditions dynamic-conditions))
            remainder (body/scalar-expression :rem :long ['inclusive-flat 'D])
            remainder-info ((ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!)
                            remainder (:values dynamic-replay))
            scalar-upper (body/->ScalarCompute (body/value 'scalar-inclusive-upper :long) remainder)
            scalar-header (assoc dynamic-header :upper 'scalar-inclusive-upper)
            scalar-kernel (assoc dynamic-kernel :operations
                                 [(assoc dynamic-guard :operations [scalar-upper scalar-header])])]
        (is (= {:lower 0 :upper (dec Long/MAX_VALUE)} (:range remainder-info)))
        (is (= (set dynamic-conditions)
               (set (get-in remainder-info [:conditional-integer :requirements]))))
        (is (some? (body/entered-for-loop-info (wide 0) remainder-info 1 :inclusive)))
        (let [scalar-info (ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!)
              maximum (scalar-info (body/scalar-expression :quot :long
                                      [(body/literal Long/MAX_VALUE :long) (body/literal 1 :long)])
                                   (:values dynamic-replay))
              signed (scalar-info (body/scalar-expression :quot :long
                                     [(body/literal -1 :long) (body/literal 1 :long)])
                                  (:values dynamic-replay))
              outside (:values (body/initial-scheduled-index-info
                                {:parameters dynamic-parameters :indices indices :launch (:launch region)}))
              unknown (scalar-info (body/scalar-expression :quot :long ['N 'D]) outside)]
          (is (= {:lower Long/MAX_VALUE :upper Long/MAX_VALUE} (:range maximum)))
          (is (nil? (body/entered-for-loop-info (wide 0) maximum 1 :inclusive)))
          (is (= {:lower -1 :upper -1} (:range signed)))
          (is (nil? (body/entered-for-loop-info (wide 0) signed 1 :inclusive)))
          (is (nil? (:conditional-integer unknown)))
          (is (nil? (body/entered-for-loop-info (wide 0) unknown 1 :inclusive))))
        (is (= scalar-kernel (body/validate! scalar-kernel))
            "actual ScalarCompute remainder certifies the inclusive exiting increment")
        (let [quotient-expression (body/scalar-expression :quot :long ['scalar-inclusive-upper 'D])
              values (assoc (:values dynamic-replay) 'scalar-inclusive-upper remainder-info)
              quotient-info ((ns-resolve 'raster.compiler.ir.kernel-body 'scalar-info!) quotient-expression values)
              quotient-compute (body/->ScalarCompute (body/value 'scalar-quotient-upper :long) quotient-expression)
              quotient-kernel (assoc dynamic-kernel :operations
                                    [(assoc dynamic-guard :operations
                                            [scalar-upper quotient-compute
                                             (assoc dynamic-header :upper 'scalar-quotient-upper)])])]
          (is (= {:lower 0 :upper (dec Long/MAX_VALUE)} (:range quotient-info)))
          (is (= (set dynamic-conditions)
                 (set (get-in quotient-info [:conditional-integer :requirements]))))
          (is (some? (body/entered-for-loop-info (wide 0) quotient-info 1 :inclusive)))
          (is (= quotient-kernel (body/validate! quotient-kernel)))
          (doseq [condition dynamic-conditions]
            (is (= :kernel-body-index-precondition
                   (try (body/validate! (assoc-in quotient-kernel [:attributes :required-preconditions]
                                                 (vec (remove #{condition} dynamic-conditions)))) :accepted
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
        (doseq [condition dynamic-conditions]
          (is (= :kernel-body-index-precondition
                 (try (body/validate! (assoc-in scalar-kernel [:attributes :required-preconditions]
                                               (vec (remove #{condition} dynamic-conditions)))) :accepted
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
              "numeric refinement never removes an executed child/domain requirement")))
      (is (precondition/check! dynamic-conditions {'N 1 'D 1}))
      (doseq [divisor [0 -1]]
        (is (precondition/check! dynamic-conditions {'N 0 'D divisor}))
        (is (thrown? clojure.lang.ExceptionInfo
              (precondition/check! dynamic-conditions {'N 1 'D divisor}))))
      (doseq [condition dynamic-conditions]
        (is (= :kernel-body-index-precondition
               (try (body/validate! (assoc-in dynamic-kernel [:attributes :required-preconditions]
                                             (vec (remove #{condition} dynamic-conditions)))) :accepted
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "actual dynamic division needs its independently enforced child/domain conditions")))))
