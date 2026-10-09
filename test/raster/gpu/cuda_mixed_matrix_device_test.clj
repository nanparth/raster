(ns raster.gpu.cuda-mixed-matrix-device-test
  "Public Float storage with explicitly requested f16-product/f32 matrix scheduling."
  (:require [clojure.test :refer [deftest is]]
            [clojure.string :as string]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.value :as value]
            [raster.hardware-fixture :as fixture]
            [raster.perf.production-canary :as canary]
            [raster.runtime.hardware :as hardware]))

(deftm independent-dyadic-reference
  [A :- (Array float) B :- (Array float)] :- (Array float)
  ;; Independent host coordinate loops, no Raster BLAS/contraction operation.
  ;; Every input is exactly half-representable; all 64-term sums are exact in f32.
  (let [result (float-array 4096)]
    (dotimes [i 64]
      (dotimes [j 64]
        (aset result (+ (* i 64) j)
              (float (loop [k 0 total 0.0]
                       (if (= k 64) total
                           (recur (inc k)
                                  (+ total (* (double (aget A (+ (* i 64) k)))
                                              (double (aget B (+ (* k 64) j))))))))))))
    result))

(defn- inputs [step]
  {:A (float-array (map #(float (/ (- (mod (+ % (* 3 step)) 17) 8) 4.0)) (range 4096)))
   :B (float-array (map #(float (/ (- (mod (+ (* 5 %) (* 7 step)) 19) 9) 4.0)) (range 4096)))})

(deftest dyadic-inputs-justify-exact-mixed-matrix-reference
  (let [first-inputs (inputs 0) changed (inputs 1)
        first-result (independent-dyadic-reference (:A first-inputs) (:B first-inputs))
        changed-result (independent-dyadic-reference (:A changed) (:B changed))]
    (doseq [dataset [first-inputs changed] key [:A :B]]
      (is (every? #(= (float %) (Float/float16ToFloat (Float/floatToFloat16 (float %))))
                  (get dataset key))))
    (is (= 4096 (alength first-result) (alength changed-result)))
    (is (every? #(Float/isFinite (float %)) first-result))
    (is (not= (vec first-result) (vec changed-result)))))

(deftest direct-matrix-request-retains-half-storage-boundary
  (fixture/isolated
   (fn []
     ;; Registered compiler fixture, not a device probe or a performance claim.
     (with-redefs [hardware/init! (fn [] nil)]
       (hardware/register-target-device!
        :cuda:0 {:name "CUDA observed-limit mixed-matrix compiler fixture"
                 :capabilities {:compute-capability [8 6] :sm-count 40 :warp-size 32
                                :max-threads-per-block 1024 :max-warps-per-sm 48
                                :max-blocks-per-sm 16 :shared-memory-per-block 49152}})
       (let [{:keys [A B]} (inputs 0)
             C (float-array (concat (repeat 4096 123.0) [-4096.0 -4096.0 -4096.0]))
             options {:compiler :equation-first :target :cuda:0 :dtype :float
                      :outputs '[C] :roles {'C :state} :on-non-resident :throw}
             control (compiled/lower #'canary/gemm64! [A B C]
                                     (assoc options :schedule {:precision :f32-scalar}))
             attempt (try
                       {:prepared (compiled/lower
                                   #'canary/gemm64! [A B C]
                                   (assoc options :gemm-precision :mixed-f16-f32
                                          :schedule {:precision :mixed-f16-f32
                                                     :typed-contraction {:strategy :matrix}}))}
                       (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
         (is (compiled/prepared? control) "ordinary FP32 public route is a working control")
         (println "[PUBLIC CUDA MIXED ADMISSION]" (select-keys attempt [:error]))
         ;; This direct single-body strategy intentionally requires half storage facts.
         ;; Float storage belongs to the existing consented multistage dispatch below.
         (is (= :kernel-graph-contraction-schedule (get-in attempt [:error :reason])))
         (is (seq (get-in attempt [:error :schedule-decline :declines])))
         (is (every? #(= :dtype-not-dpas (:reason %))
                     (get-in attempt [:error :schedule-decline :declines]))))))))

(deftest non-half-representable-ties-distinguish-conversion-from-fp32-fallback
  (doseq [[raw expected] [[1.00048828125 1.0]
                          [1.00146484375 1.001953125]
                          [-1.00048828125 -1.0]
                          [-1.00146484375 -1.001953125]]]
    (is (= (float expected) (Float/float16ToFloat (Float/floatToFloat16 (float raw))))
        "independent explicit half tie chooses the even significand"))
  (let [raw-a (float-array (repeat 4096 (float 1.00048828125)))
        rounded-a (float-array (repeat 4096 (float 1.0)))
        b (float-array (repeat 4096 (float 1.0)))
        ordinary (independent-dyadic-reference raw-a b)
        mixed (independent-dyadic-reference rounded-a b)]
    (is (every? #(= (float 64.0) %) mixed))
    (is (every? #(= (float 64.03125) %) ordinary))
    (is (not= (vec ordinary) (vec mixed))
        "exactly representable outputs discriminate missing f32-to-f16 conversion")))

(deftest public-cuda-materialized-mixed-dispatch-reaches-mma
  (fixture/isolated
   (fn []
     (with-redefs [hardware/init! (fn [] nil)]
       (hardware/register-target-device!
        :cuda:0 {:name "CUDA observed-limit mixed-dispatch compiler fixture"
                 :capabilities {:compute-capability [8 6] :sm-count 40 :warp-size 32
                                :max-threads-per-block 1024 :max-warps-per-sm 48
                                :max-blocks-per-sm 16 :shared-memory-per-block 49152}})
       (let [{:keys [A B]} (inputs 0)
             C (float-array (concat (repeat 4096 123.0) [-4096.0 -4096.0 -4096.0]))
             prepared (compiled/lower
                       #'canary/gemm64! [A B C]
                       {:compiler :equation-first :target :cuda:0 :dtype :float
                        :outputs '[C] :roles {'C :state} :on-non-resident :throw
                        :schedule {:precision :mixed-f16-f32
                                   :typed-contraction {:strategy :dispatch-mixed-matrix
                                                       :input-fusion :materialized
                                        :tile {:block-m 64 :block-n 64 :block-k 32
                                               :sg-m 32 :sg-n 32 :num-stages 3
                                               :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32}}}}})
             parts (tree-seq coll? seq (compiled/plan prepared))
             models (filter #(and (map? %) (= :mixed-matrix-operational-model (:kind %))) parts)
             sources (keep #(when (and (map? %) (string? (:source %))) (:source %)) parts)]
         (is (compiled/prepared? prepared))
         (println "[PUBLIC CUDA MATERIALIZED MIXED]" {:models (count models)
                                                      :wmma-sources (count (filter #(string/includes? % "wmma::mma_sync") sources))})
         (is (seq models) "public dispatch retains the independently reconstructed conversion/accumulation model")
         (is (some #(string/includes? % "wmma::mma_sync") sources)
             "public mixed candidate emits existing WMMA rather than only its portable FP32 fallback"))))))


(def mixed-options
  {:compiler :equation-first :target :cuda:0 :dtype :float
   :outputs '[C] :roles {'C :state} :on-non-resident :throw
   :schedule {:precision :mixed-f16-f32
              :typed-contraction {:strategy :dispatch-mixed-matrix :input-fusion :materialized
                                  :tile {:block-m 64 :block-n 64 :block-k 32
                                         :sg-m 32 :sg-n 32 :num-stages 3
                                         :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32}}}}})

(defn- bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- rounding-inputs []
  {:A (float-array (repeat 4096 (float 1.00048828125)))
   :B (float-array (repeat 4096 (float 1.0)))})

(defn- prove-public-mixed-replay! []
  (let [{:keys [A B]} (inputs 0)
        C (float-array (concat (repeat 4096 123.0) (repeat 3 -4096.0)))
        prepared (compiled/lower #'canary/gemm64! [A B C] mixed-options)
        program (compiled/instantiate! prepared)
        last-output (atom nil)]
    (try
      (let [parts (tree-seq coll? seq (compiled/plan prepared))]
        (is (some #(and (map? %) (= :mixed-matrix-operational-model (:kind %))) parts))
        (is (some #(and (map? %) (string? (:source %))
                        (string/includes? (:source %) "wmma::mma_sync")) parts)))
      (let [info (compiled/execution-info program)
            descriptions (filter #(and (map? %) (= :xmx-direct (:strategy %)))
                                 (tree-seq coll? seq info))]
        (println "[CUDA MIXED DEVICE]" (runtime/execution-device-info))
        (println "[CUDA MIXED BOUND EXECUTION]" info)
        (is (seq descriptions) "existing shared materialized mixed strategy is actually bound")
        (is (some #(= :mixed-f16-f32 (:precision %)) descriptions))
        (is (some #(= 3 (count (:entry-points %))) descriptions)
            "bound conversionA/conversionB/matrix entry points precede completed replay"))
      (doseq [current [(inputs 0) (inputs 1) (rounding-inputs)]]
        (let [prior @last-output
              output (:C (program current))
              actual (value/->host output)
              expected (if (= (float 1.00048828125) (aget (:A current) 0))
                         (float-array (repeat 4096 (float 64.0)))
                         (independent-dyadic-reference (:A current) (:B current)))
              session (get-in program [:executable :session])
              key (first (for [[key buffer] (:buffers @session)
                               :when (identical? (cleanup/lifetime-owner buffer)
                                                 (cleanup/lifetime-owner (:buffer output)))] key))
              physical (float-array 4099)]
          (reset! last-output output)
          (when prior (is (not (value/live? prior))))
          ;; Public C names the complete caller storage; its logical produced prefix is4096.
          ;; This is the same full-state return convention as accepted NN/NT matrix tests.
          (is (= 4099 (alength actual)))
          (is (= (bits (concat expected (repeat 3 -4096.0))) (bits actual)))
          (println "[CUDA MIXED COMPLETED CASE]"
                   {:raw-rounding-tie? (= (float 1.00048828125) (aget (:A current) 0))
                    :first-output (aget actual 0) :logical-prefix-elements 4096
                    :returned-elements (alength actual) :guards (vec (drop 4096 actual))})
          (is (= 16396 (:byte-size (gpu/buffer session key))))
          (gpu/download-range! session (gpu/buffer-view session key {:shape [4099]})
                               physical {:elements 4099})
          (is (= (bits expected) (bits (take 4096 physical))))
          (is (= (bits (repeat 3 -4096.0)) (bits (drop 4096 physical))))))
      (finally (compiled/close! program)))
    (is (not (value/live? @last-output)))
    (is (zero? (root/lease-count runtime/state)))))

(deftest required-public-cuda-mixed-matrix-converts-raw-inputs-and-replays
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (println "[CUDA SKIP] mixed qualification requires RASTER_REQUIRE_CUDA_DEVICE=1")
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-public-mixed-replay!)))))

(defn- padded [offset values]
  (float-array (concat (repeat offset 4096.0) values (repeat 3 -4096.0))))

(defn- prove-generated-mixed-offsets! []
  (let [{:keys [A B]} (inputs 0)
        prepared (compiled/lower #'canary/gemm64! [A B (float-array 4099)] mixed-options)
        step (get-in (compiled/plan prepared) [:instances 0 :call :steps 0])
        graph (:graph step)
        scalars (select-keys (:scalar-values step) (map :id (:scalars graph)))
        session (gpu/make-session :cuda:0)]
    (try
      (gpu/alloc! session {:A [:float 4107 nil] :B [:float 4115 nil] :C [:float 4107 nil]})
      (let [a-view (gpu/buffer-view session :A {:byte-offset 32 :shape [4096]})
            b-view (gpu/buffer-view session :B {:byte-offset 64 :shape [4096]})
            c-view (gpu/buffer-view session :C {:byte-offset 32 :shape [4096]})
            misaligned (gpu/buffer-view session :C {:byte-offset 4 :shape [4096]})
            reason (try (gpu/bind-kernel-graph! session :misaligned-mixed graph
                                              {'A a-view 'B b-view 'C misaligned} scalars)
                        :accepted (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))]
        (is (= :kernel-abi-pointer-alignment reason))
        (let [bound (gpu/bind-kernel-graph! session :aligned-mixed graph
                                           {'A a-view 'B b-view 'C c-view} scalars)]
          (is (= [32 64 32] (mapv #(get-in % [:view :byte-offset]) [a-view b-view c-view])))
          (doseq [current [(inputs 0) (inputs 1) (rounding-inputs)]]
            (let [a-owner (padded 8 (:A current)) b-owner (padded 16 (:B current))
                  c-owner (padded 8 (repeat 4096 123.0))
                  expected (if (= (float 1.00048828125) (aget (:A current) 0))
                             (repeat 4096 (float 64.0))
                             (independent-dyadic-reference (:A current) (:B current)))]
              (gpu/upload! session :A a-owner)
              (gpu/upload! session :B b-owner)
              (gpu/upload! session :C c-owner)
              (with-redefs [value/->host (fn [& _] (throw (ex-info "mixed graph host materialization" {})))]
                (gpu/run-kernel-graph! session bound))
              (is (= (bits a-owner) (bits (gpu/download session :A))))
              (is (= (bits b-owner) (bits (gpu/download session :B))))
              (is (= (bits (padded 8 expected)) (bits (gpu/download session :C))))))))
      (finally (gpu/close-session! session)))
    (is (zero? (root/lease-count runtime/state)))))

(deftest required-generated-mixed-graph-preserves-aligned-pointer-views-and-guards
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (println "[CUDA SKIP] mixed offset qualification requires RASTER_REQUIRE_CUDA_DEVICE=1")
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-generated-mixed-offsets!)))))
