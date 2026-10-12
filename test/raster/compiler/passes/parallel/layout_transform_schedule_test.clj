(ns raster.compiler.passes.parallel.layout-transform-schedule-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.backend.gpu.layout-transform :as layout-emitter]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-precondition :as precondition]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]))

(deftest cast-is-a-typed-unrolled-affine-kernel-body
  (let [kernel (schedule/cast-body
                {:id :cast :input 'x :output 'y :extent-id 'n
                 :source-dtype :float :destination-dtype :half :vector-width 4
                 :rounding :nearest-even :overflow :ieee})]
    (is (body/kernel-body? kernel))
    (is (= [:float :half :int] (mapv :dtype (:parameters kernel))))
    (is (= 4 (count (:masks kernel))))
    (is (= 12 (count (:operations kernel))))
    (is (= [2] (:group-count
                (launch/realize (:launch kernel) {'n 1025})))
        "the masked tail owns a work-item even when the extent is not divisible by four")
    (is (= {:strategy :affine-elementwise-cast
            :elements-per-work-item 4 :rounding :nearest-even :overflow :ieee}
           (:schedule kernel)))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (testing (name dialect)
        (let [source (emitter/emit-scalar-kernel "cast_rows" kernel
                                                 {:target-dialect dialect})]
          (is (str/includes? source "cast_rows"))
          (is (str/includes? source "layout_converted_3")))))))

(deftest transpose-is-an-affine-permutation-not-target-source
  (let [kernel (schedule/transpose-body
                {:id :transpose :input 'x :output 'y
                 :row-extent-id 'm :column-extent-id 'n
                 :element-dtype :half})
        module (emitter/emit-scalar-module "transpose_rows" kernel)
        source (:source module)]
    (is (body/kernel-body? kernel))
    (is (= [1 0] (get-in kernel [:schedule :permutation])))
    (is (= [:layout-row :layout-column]
           (get-in kernel [:operations 0 :coordinates])))
    (is (= [:layout-column :layout-row]
           (get-in kernel [:operations 1 :coordinates])))
    (is (str/includes? source "rstr_layout_column"))
    (is (str/includes? source "rstr_layout_row"))))

(deftest layout-extents-preserve-independent-integer-widths
  (doseq [extent-dtype [:int :long]]
    (let [kernel (layout-emitter/cast-body
                  {:id :wide-cast :input 'x :output 'y :extent-dtype extent-dtype
                   :source-dtype :float :destination-dtype :half :vector-width 4
                   :rounding :nearest-even :overflow :ieee})]
      (is (= extent-dtype (:dtype (last (:parameters kernel)))))
      (is (= [2] (:group-count (launch/realize (:launch kernel) {:layout-elements 1025}))))
      (doseq [dialect [:opencl-portable :cuda :hip]]
        (is (string? (:source (layout-emitter/emit-cast-kernel
                               {:kernel-name "wide_cast" :input 'x :output 'y
                                :extent-dtype extent-dtype
                                :source-dtype :float :destination-dtype :half
                                :vector-width 4 :rounding :nearest-even :overflow :ieee
                                :target-dialect dialect})))))))
  (doseq [row-dtype [:int :long] column-dtype [:int :long]]
    (let [options {:id :wide-transpose :input 'x :output 'y :element-dtype :float
                   :row-extent-dtype row-dtype :column-extent-dtype column-dtype}
          kernel (layout-emitter/transpose-body options)]
      (is (= [row-dtype column-dtype] (mapv :dtype (drop 2 (:parameters kernel)))))
      (doseq [dialect [:opencl-portable :cuda :hip]]
        (is (string? (:source (layout-emitter/emit-transpose-kernel
                               (assoc options :kernel-name "wide_transpose"
                                      :target-dialect dialect)))))))))

(deftest layout-extents-reject-non-index-representations
  (doseq [invalid [:float :double :half]
          make-kernel [(fn [] (schedule/cast-body
                               {:input 'x :output 'y :extent-dtype invalid
                                :source-dtype :float :destination-dtype :half
                                :rounding :nearest-even :overflow :ieee}))
                       (fn [] (schedule/transpose-body
                               {:input 'x :output 'y :element-dtype :float
                                :row-extent-dtype invalid}))
                       (fn [] (schedule/transpose-body
                               {:input 'x :output 'y :element-dtype :float
                                :column-extent-dtype invalid}))]]
    (try
      (make-kernel)
      (is false "non-index extent dtype must fail before emission")
      (catch clojure.lang.ExceptionInfo e
        (is (= :layout-transform-index-dtype (:reason (ex-data e))))))))

(deftest long-extents-have-checked-non-allocating-launch-boundaries
  (let [cast (schedule/cast-body
              {:id :large-cast :input 'x :output 'y :extent-dtype :long
               :source-dtype :float :destination-dtype :half :vector-width 4
               :rounding :nearest-even :overflow :ieee})
        transpose (schedule/transpose-body
                   {:id :large-transpose :input 'x :output 'y :element-dtype :float
                    :row-extent-dtype :long :column-extent-dtype :long})]
    (is (= [2097152]
           (:group-count (launch/realize (:launch cast) {:layout-elements 2147483648})))
        "a logical extent above int range is retained without allocating that buffer")
    (is (thrown? ArithmeticException
                 (launch/realize (:launch transpose)
                                 {:layout-rows 3037000500 :layout-cols 3037000500}))
        "shape products cannot wrap the checked host launch arithmetic")))


(deftest generated-layout-prefixes-widen-before-every-hardware-operation
  (doseq [extent-dtype [:int :long] vector-width [1 2 4]]
    (let [kernel (schedule/cast-body
                  {:id :prefix-cast :input 'x :output 'y :extent-dtype extent-dtype
                   :source-dtype :float :destination-dtype :half :vector-width vector-width
                   :rounding :nearest-even :overflow :ieee})
          global (get-in kernel [:indices 2 :expression])
          expected (body/expression
                    :add
                    (body/expression :mul
                                     (body/index-cast :layout-group :long :exact)
                                     (body/index-cast 256 :long :exact))
                    (body/index-cast :layout-lane :long :exact))
          env {:layout-group Integer/MAX_VALUE :layout-lane 255}
          work-item (launch/resolve-expression env global)
          base (launch/resolve-expression
                (assoc env :layout-work-item work-item)
                (get-in kernel [:indices 3 :expression]))]
      (is (= expected global))
      (is (= 549755813887 work-item)
          "algebraic hardware-coordinate evaluation, without allocation or device launch")
      (let [accepted (launch/geometry
                      {:workgroup-size [256]
                       :group-count [(inc (long Integer/MAX_VALUE))]})
            rejected (launch/geometry
                      {:workgroup-size [256]
                       :group-count [(+ 2 (long Integer/MAX_VALUE))]})]
        (is (= accepted (body/validate-launch-index-ranges! kernel accepted)))
        (try
          (body/validate-launch-index-ranges! kernel rejected)
          (is false "Long synthetic arithmetic does not widen physical Int group bindings")
          (catch clojure.lang.ExceptionInfo e
            (is (= :kernel-launch-index-range (:reason (ex-data e)))))))
      (is (> work-item Integer/MAX_VALUE))
      (is (= (* work-item vector-width) base))
      (is (= extent-dtype (:dtype (last (:parameters kernel)))))
      (doseq [offset (range vector-width)]
        (is (= (+ base offset)
               (launch/resolve-expression
                {:layout-base base}
                (first (:coordinates (nth (:operations kernel) (* offset 3))))))))))
  (is (thrown? clojure.lang.ExceptionInfo
               (launch/geometry {:workgroup-size [256]
                                 :group-count [0]})))
  (let [kernel (schedule/transpose-body
                {:id :prefix-transpose :input 'x :output 'y :element-dtype :float})]
    (is (= :add (get-in kernel [:indices 2 :expression :op])))
    (is (every? #(= :long (:dtype %))
                [(get-in kernel [:indices 2 :expression :arguments 0 :arguments 0])
                 (get-in kernel [:indices 2 :expression :arguments 0 :arguments 1])
                 (get-in kernel [:indices 2 :expression :arguments 1])]))))


(deftest transpose-totalized-coordinates-retain-a-checked-shape-invocation
  (doseq [row-type [:int :long] col-type [:int :long]]
    (let [options {:kernel-name "transpose_domain" :input 'x :output 'y :element-dtype :float
                   :row-extent-dtype row-type :column-extent-dtype col-type
                   :target-dialect :opencl-portable}
          emitted (layout-emitter/emit-transpose-artifact options)
          kernel (artifact/attribute emitted :kernel-body)
          conditions (body/required-preconditions kernel)
          arguments (fn [rows cols] [:source-buffer :destination-buffer
                                    {:type row-type :value rows} {:type col-type :value cols}])]
      (is (= conditions (:preconditions emitted)))
      (is (= [row-type col-type] (mapv :dtype (drop 2 (:parameters kernel)))))
      (is (call/kernel-call? (call/make emitted (arguments 3 4))))
      (doseq [[rows cols] [[-1 4] [3 -1] [-2 -3]]]
        (try
          (call/make emitted (arguments rows cols))
          (is false "each negative logical axis rejects before allocation or launch")
          (catch clojure.lang.ExceptionInfo e
            (is (= :kernel-precondition-failed (:reason (ex-data e)))))))
      (doseq [[rows cols] [[0 3] [4 0] [0 0]]]
        (is (true? (precondition/check! conditions {:layout-rows rows :layout-cols cols})))
        (is (zero? (launch/resolve-expression
                    {:layout-rows rows :layout-cols cols}
                    (first (get-in kernel [:launch :group-count])))))
        (is (thrown? clojure.lang.ExceptionInfo (call/make emitted (arguments rows cols)))
            "the existing zero launch geometry policy is retained"))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"discard"
                           (emitter/emit-scalar-kernel "source_only_transpose" kernel)))
      (let [row-expr (get-in kernel [:indices 3 :expression])
            col-expr (get-in kernel [:indices 4 :expression])]
        (doseq [cols [0 1 4] linear [0 3 11]]
          (let [env {:layout-linear linear :layout-cols cols} divisor (max cols 1)]
            (is (= (quot linear divisor) (launch/resolve-expression env row-expr)))
            (is (= (rem linear divisor) (launch/resolve-expression env col-expr))))))))
  (let [emitted (layout-emitter/emit-transpose-artifact
                 {:kernel-name "long_transpose_domain" :input 'x :output 'y :element-dtype :float
                  :row-extent-dtype :long :column-extent-dtype :long :target-dialect :opencl-portable})]
    (doseq [options [{} {:resolve-value (constantly 0)}]]
      (is (thrown? ArithmeticException
                   (call/make emitted [:source-buffer :destination-buffer
                                       {:type :long :value Long/MAX_VALUE}
                                       {:type :long :value 2}] options))))))
