(ns raster.gpu.cuda-quantized-projection-device-test
  "Existing expanded Q4_K projection on checked Int32/Byte/Float public leaves.
   The bounded dyadic data extends the independently reviewed hand-calculated fixture;
   it does not qualify arbitrary quantizers, model logits or native floating lane policy."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.test :refer [deftest is]]
            [raster.quant.kernels-k :as kernels]
            [raster.quant.pack :as pack]
            [raster.compiler.backend.cpu.quant :as quant]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.value :as value]
            [raster.gpu.compiled :as compiled]
            [raster.hardware-fixture :as fixture]
            [raster.runtime.hardware :as hardware]))

(defn dyadic-fixture []
  ;; Reviewed hand fixture per block: dot48, sum32, da=.5, aq2, db=.25,
  ;; bq=-4, xs=.5 ->64. Repeat two blocks, vary output da and signed row codes.
  ;; Every intermediate is an exactly representable bounded dyadic.
  {:xp (int-array (concat (repeat 128 0x01010101) (repeat 128 -1) (repeat 128 0)))
   :xs (float-array (repeat 6 0.5))
   :bsums (int-array (concat (repeat 16 32) (repeat 16 -32) (repeat 16 0)))
   :wp (int-array (repeat 704 0x21212121))
   :da (float-array (for [output (range 11) _ (range 2)] (* 0.5 (inc output))))
   :db (float-array (repeat 22 0.25))
   :aq (byte-array (repeat 176 2))
   :bq (byte-array (repeat 176 -4))
   :y (float-array (concat (repeat 33 123.0) (repeat 3 -4096.0)))})

(defn- arguments [{:keys [xp xs bsums wp da db aq bq y]}]
  [xp xs bsums wp da db aq bq y (long 512) (long 11) (long 3)])

(defn- expected []
  ;; Same independent closed dyadic algebra as the reviewed Python dot verification.
  (mapv #(float %) (for [row-code [1 -1 0] output (range 11)]
                    (* row-code (+ (* 384 output) 128)))))

(deftest reviewed-dyadic-packed-data-preserves-the-existing-projection-contract
  (let [compatibility (dyadic-fixture) product (dyadic-fixture)]
    (apply kernels/qmatmul-q4k-dp4a-rows! (arguments compatibility))
    (apply kernels/qmatmul-q4k-product-rows! (arguments product))
    (is (= (concat (expected) (repeat 3 (float -4096.0))) (vec (:y compatibility))))
    (is (= (vec (:y compatibility)) (vec (:y product))))
    (is (= 36 (alength (:y product))))))

(deftest public-cuda-product-projection-retains-declared-storage-and-ordered-consumer
  (fixture/isolated
   (fn []
     (with-redefs [hardware/init! (fn [] nil)]
       ;; Compiler fixture with observed CUDA limits; no device discovery/execution.
       (hardware/register-target-device!
        :cuda:0 {:name "CUDA quantized projection compiler fixture"
                 :capabilities {:compute-capability [8 6] :sm-count 40 :warp-size 32
                                :max-threads-per-block 1024 :max-warps-per-sm 48
                                :max-blocks-per-sm 16 :shared-memory-per-block 49152}})
       (let [attempt (try
                       {:prepared (compiled/lower
                                   #'kernels/qmatmul-q4k-product-rows!
                                   (arguments (dyadic-fixture))
                                   {:compiler :equation-first :target :cuda:0 :dtype :float
                                    :preserve-declared-array-storage? true
                                    :schedule {:precision :f32-scalar}
                                    :outputs '[y] :roles '{y :output}
                                    :constants '[wp da db aq bq] :on-non-resident :throw})}
                       (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
         (println "[PUBLIC CUDA Q4_K PROJECTION ADMISSION]" (select-keys attempt [:error]))
         (is (nil? (:error attempt)) "existing product equation must reach a checked public invocation")
         (when-let [prepared (:prepared attempt)]
           (is (compiled/prepared? prepared))
           (let [plan (compiled/plan prepared)
                 entries (concat (:in-tree prepared) (:out-tree prepared))
                 storage (into {} (for [{:keys [key node]} entries]
                                    [key (get-in plan [:nodes node :view :dtype])]))]
             (doseq [[key dtype] {:xp :int :bsums :int :wp :int :xs :float
                                  :da :float :db :float :aq :byte :bq :byte :y :float}]
               (is (= dtype (get storage key)) (str "public leaf " key " preserves its declared storage")))
             (is (seq (:instances plan)))
             (println "[PUBLIC CUDA Q4_K PROJECTION STORAGE]" storage))))))))


(defn- bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- seeded-values [elements seed]
  (let [random (java.util.Random. (long seed)) values (float-array elements)]
    (dotimes [index elements]
      (aset values index (float (- (.nextDouble random) 0.5))))
    values))

(deftest committed-seed-compatibility-keeps-its-existing-rounding-and-bound
  ;; The exact dimensions/seeds/bound from kernels_k_test/q4k-projection-shares-weights-...
  ;; This is shared-substrate compatibility, distinct from the independent dyadic reference.
  (let [x (seeded-values 1536 81) weights (seeded-values 5632 82)
        expanded (quant/quantize-weight-q4k weights quant/q4-K)
        packed (pack/q4k-of weights 11 512)
        xp (int-array 384) xs (float-array 6) sums (int-array 48) submax (float-array 48)
        compatibility (float-array 33) product (float-array 33)]
    (kernels/quant-act-q8k-rows-gpu! x xp xs sums submax 512 3)
    (let [args [xp xs sums (:wp packed) (:da packed) (:db packed) (:aq packed) (:bq packed)]]
      (apply kernels/qmatmul-q4k-dp4a-rows! (concat args [compatibility (long 512) (long 11) (long 3)]))
      (apply kernels/qmatmul-q4k-product-rows! (concat args [product (long 512) (long 11) (long 3)])))
    (is (= 33 (alength product)))
    (is (= (bits compatibility) (bits product)))
    (doseq [row (range 3)]
      (let [row-x (float-array 512) expected-row (float-array 11)]
        (System/arraycopy x (* row 512) row-x 0 512)
        (let [{:keys [xq xs bsums]} (quant/quantize-act-q8k row-x 512 quant/q4-K)]
          (kernels/qmatmul-q4k-composable! xq xs bsums
                                           (:wq expanded) (:da expanded) (:db expanded)
                                           (:aq expanded) (:bq expanded) expected-row 512 11 0 11))
        (doseq [output (range 11)]
          (is (< (Math/abs (- (double (aget product (+ (* row 11) output)))
                             (double (aget expected-row output)))) 1e-3)))))))

(def projection-options
  {:compiler :equation-first :target :cuda:0 :dtype :float
   :preserve-declared-array-storage? true :schedule {:precision :f32-scalar}
   :outputs '[y] :roles '{y :output} :constants '[wp da db aq bq] :on-non-resident :throw})

(defn- changed-dyadic-inputs [codes]
  {:xp (int-array (mapcat #(repeat 128 (case % 1 0x01010101 -1 -1 0 0)) codes))
   :xs (float-array (repeat 6 0.5))
   :bsums (int-array (mapcat #(repeat 16 (* 32 %)) codes))})

(defn- native-projection! []
  (let [prepared (compiled/lower #'kernels/qmatmul-q4k-product-rows!
                                 (arguments (dyadic-fixture)) projection-options)
        program (compiled/instantiate! prepared) last-output (atom nil)]
    (try
      (let [info (compiled/execution-info program)
            descriptions (filter #(and (map? %) (contains? % :entry-points))
                                 (tree-seq coll? seq info))]
        (println "[CUDA Q4_K PROJECTION BOUND]" info)
        (println "[CUDA Q4_K PROJECTION DEVICE]" (runtime/execution-device-info))
        (is (some #(= 1 (count (:entry-points %))) descriptions))
        (is (= 1 (get-in prepared [:schedule :stats :emission :product-consumer-regions-emitted]))))
      (doseq [codes [[1 -1 0] [-1 1 0] [0 0 0]]]
        (let [previous @last-output
              output (:y (program (changed-dyadic-inputs codes)))
              actual (value/->host output)
              expected (for [code codes column (range 11)]
                         (float (* code (+ (* 384 column) 128))))]
          (reset! last-output output)
          (when previous (is (not (value/live? previous))))
          (is (= 36 (alength actual)))
          (is (= (bits (concat expected (repeat 3 -4096.0))) (bits actual)))
          (println "[CUDA Q4_K PROJECTION COMPLETED]"
                   {:codes codes :logical-elements 33 :returned-elements (alength actual)
                    :first-output (aget actual 0) :guards (vec (drop 33 actual))})))
      (finally (compiled/close! program)))
    (is (not (value/live? @last-output)))
    (is (zero? (root/lease-count runtime/state)))))

(deftest required-cuda-product-projection-executes-and-replays-checked-packed-inputs
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "projection qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (native-projection!)))))
