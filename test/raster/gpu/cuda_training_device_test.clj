(ns raster.gpu.cuda-training-device-test
  "Portable tiny linear/MSE/SGD reference and opt-in resident CUDA donation proof."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as ad]
            [raster.arrays :as arrays]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.core :refer [deftm]]
            [raster.dl.loss :as loss]
            [raster.dl.nn :as nn]
            [raster.dl.optim :as optim]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.value :as value]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.runtime.hardware :as hardware]))

;; Faithful standalone definitions from gpu-ad-full-train-step-fully-resident,
;; gpu_ad_gemm_test.clj SHA256 86ecbea1dc811178c626e7b2650b066a63eb079f9b8ed6e4c874c9fd5edf82b6.
;; That fixture defines these inside its deftest; do not invoke its unrelated GPU tests.
(deftm tiny-training-loss
  [W :- (Array float) x :- (Array float) tgt :- (Array float)
   batch :- Long in-f :- Long out-f :- Long] :- Double
  (let [pred (nn/linear-nb x W batch in-f out-f)]
    (loss/mse-loss pred tgt (clojure.core/* batch out-f))))

(deftm tiny-training-step
  [W :- (Array float) x :- (Array float) tgt :- (Array float)
   batch :- Long in-f :- Long out-f :- Long lr :- Double] :- (Array float)
  (let [vg ((ad/value+grad #'tiny-training-loss) W x tgt batch in-f out-f)
        dW (clojure.core/nth vg 1)]
    (optim/sgd-step! W dW (arrays/alength W) (float lr))
    W))

(deftm tiny-training-weight-gradient
  [W :- (Array float) x :- (Array float) tgt :- (Array float)
   batch :- Long in-f :- Long out-f :- Long] :- (Array float)
  (clojure.core/nth ((ad/value+grad #'tiny-training-loss) W x tgt batch in-f out-f) 1))

(deftm tiny-training-mse-primal
  [pred :- (Array float) target :- (Array float) n :- Long] :- Double
  (loss/mse-loss pred target n))

;; Supplemental exact signed-dyadic cases, generator/reference independently
;; prepared without Raster/BLAS; reference.json SHA256
;; 8669df85f8c5a35a0b6639e09833c96c01be94d2e6db35ef9d5241f94b0fea50.
;; All enumerated products/sum associations/residuals/scalings are exact f32.
;; This does not certify the seeded gradients or general FP32 association.
(def dyadic-input [1/2 -1 3/2 -3/4 1/4 1])
(def dyadic-target [1/4 -1/2 1 -1/4])
(def dyadic-cases
  [{:W [1 -1/2 1/4 -1 3/4 1/2]
    :loss 1729/1024 :dW [57/64 -49/64 1/32 -81/128 27/128 27/32]}
   {:W [3/4 -1/4 1/2 -1/2 1 -3/4]
    :loss 97/64 :dW [45/64 -45/64 9/32 -33/64 61/64 -43/32]}])

(deftest public-primal-loss-retains-target-and-declared-double-output
  ;; Compiler-only synthetic descriptor; this assertion executes no CUDA kernel.
  (hardware-fixture/isolated
   (fn []
     (hardware/register-target-device!
      :cuda:0 {:name "synthetic training compiler contract"
               :capabilities {:compute-capability [8 6] :sm-count 40 :warp-size 32
                              :max-threads-per-block 1024 :max-warps-per-sm 48
                              :max-blocks-per-sm 16 :shared-memory-per-block 49152}})
     (let [prepared (compiled/lower
                     #'tiny-training-loss
                     [(float-array (:W (first dyadic-cases)))
                      (float-array dyadic-input) (float-array dyadic-target)
                      (long 2) (long 3) (long 2)]
                     {:compiler :equation-first :target :cuda:0 :dtype :float
                      :constants '[x tgt]})
           plan (compiled/plan prepared)
           outputs (mapv #(get-in plan [:nodes % :view]) (:outputs plan))]
       (is (compiled/prepared? prepared))
       (is (= [:double] (mapv :dtype outputs)))
       (is (= [[]] (mapv :shape outputs)))))))

(deftest public-mse-double-division-discriminator-prepares-as-scalar-double
  (hardware-fixture/isolated
   (fn []
     (hardware/register-target-device!
      :cuda:0 {:name "synthetic MSE scalar compiler contract"
               :capabilities {:compute-capability [8 6] :sm-count 40 :warp-size 32
                              :max-threads-per-block 1024 :max-warps-per-sm 48
                              :max-blocks-per-sm 16 :shared-memory-per-block 49152}})
     (let [prepared (compiled/lower #'tiny-training-mse-primal
                                    [(float-array [1 0 0]) (float-array [0 0 0]) (long 3)]
                                    {:compiler :equation-first :target :cuda:0 :dtype :float
                                     :constants '[pred target]})
           plan (compiled/plan prepared)
           view (get-in plan [:nodes (first (:outputs plan)) :view])]
       (is (= 1 (count (:outputs plan))))
       (is (= :double (:dtype view)))
       (is (= [] (:shape view)))
       (is (= 8 (:byte-length view)))))))

;; Independent reference package controller-accepted on 2026-10-08.
;; reference.json SHA256 aca0133e7f5728cd888e72ea28d1607b00ebc368a1aa190d47a39364b083c75c.
;; Portable frozen constants only: no workbench file is read at test runtime.
;; These bits describe explicitly ordered scalar binary32, not a required GEMM association.
(def initial-bits
  {:W ["3e6cf83c" "bddb841b" "3e926576" "3e0e5fab" "3e7c6871" "3eeffca8"]
   :x ["3e69935b" "3e29f0a8" "3ec61c55" "3daaccb3" "bd64352e" "3ebb61ac"]
   :tgt ["3e697be0" "3eabc6da" "bd2f2d23" "be098ba1"]})

(def ordered-update-bits
  [["3e6cfdfd" "bddb470c" "3e9250ef" "3e0e56d6" "3e7c905a" "3eefc87a"]
   ["3e6d03c2" "bddb09fb" "3e923c6e" "3e0e4e10" "3e7cb848" "3eef945f"]])

;; Exact mathematical loss and W coordinate derivatives at initial W and at
;; the first ordered binary32 update. Target cotangents are not compared here:
;; the committed MSE AD rule freezes them; shape integers have no tangent.
(def mathematical-reference
  [{:loss 43972458607578631396895660837337/1298074214633706907132624082305024
    :dW [-10611490629090779185681/4835703278458516698824704
         -56316888043124546684451/4835703278458516698824704
         4733568044835445450359/302231454903657293676544
         4072198879274873044709/1208925819614629174706176
         -18405859980848077765555/1208925819614629174706176
         12031442812535556353345/302231454903657293676544]}
   {:loss 10985934856110262810048438619941/324518553658426726783156020576256
    :dW [-5321993871663096821369/2417851639229258349412352
         -28161545424318523430567/2417851639229258349412352
         4728020084954701951421/302231454903657293676544
         2022735168105212302775/604462909807314587353088
         -9207130860657105085477/604462909807314587353088
         6007390797215874893117/151115727451828646838272]}])

(defn- decoded-bits [hex]
  (mapv #(Integer/parseUnsignedInt % 16) hex))

(defn- decoded-floats [hex]
  (mapv #(Float/intBitsToFloat %) (decoded-bits hex)))

(defn- bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- seeded [n seed]
  (let [random (java.util.Random. (long seed))
        result (float-array n)]
    (dotimes [i n] (aset-float result i (float (- (.nextDouble random) 0.5))))
    result))

(defn- exact-float [value]
  (let [raw (Float/floatToRawIntBits (float value))
        exponent (bit-and 255 (unsigned-bit-shift-right raw 23))
        fraction (bit-and 8388607 raw)
        significand (* (if (neg? raw) -1 1)
                       (if (zero? exponent) fraction (+ 8388608 fraction)))
        shift (if (zero? exponent) -149 (- exponent 150))]
    (when (= 255 exponent)
      (throw (ex-info "the finite training reference excludes infinity and NaN" {})))
    (if (neg? shift)
      (/ (bigint significand) (bigint (.shiftLeft java.math.BigInteger/ONE (- shift))))
      (* (bigint significand) (bigint (.shiftLeft java.math.BigInteger/ONE shift))))))

(defn- coordinate-reference [W x tgt]
  ;; Exact row-major rational arithmetic, independent of Raster AD/BLAS.
  (let [pred (vec (for [b (range 2) o (range 2)]
                    (reduce + 0 (for [i (range 3)]
                                  (* (nth x (+ (* b 3) i)) (nth W (+ (* o 3) i)))))))
        residual (mapv - pred tgt)]
    {:loss (/ (reduce + 0 (map #(* % %) residual)) 4)
     :dW (vec (for [o (range 2) i (range 3)]
                (reduce + 0 (for [b (range 2)]
                              (* (/ (nth residual (+ (* b 2) o)) 2)
                                 (nth x (+ (* b 3) i)))))))}))

(defn- f32-add [a b] (float (+ (double a) (double b))))
(defn- f32-sub [a b] (float (- (double a) (double b))))
(defn- f32-mul [a b] (float (* (double a) (double b))))

(defn- ordered-update [W x tgt]
  ;; Explicitly round multiplication before addition and every scalar store.
  ;; This is the reviewed investigation reference, not a new matrix policy.
  (let [pred (vec (for [b (range 2) o (range 2)]
                    (reduce f32-add (float 0.0)
                            (for [i (range 3)]
                              (f32-mul (nth x (+ (* b 3) i)) (nth W (+ (* o 3) i)))))))
        dy (mapv #(f32-mul (f32-sub %1 %2) (float 0.5)) pred tgt)
        dW (vec (for [o (range 2) i (range 3)]
                  (reduce f32-add (float 0.0)
                          (for [b (range 2)]
                            (f32-mul (nth dy (+ (* b 2) o)) (nth x (+ (* b 3) i)))))))]
    (mapv #(f32-sub %1 (f32-mul (float 0.01) %2)) W dW)))

(deftest portable-training-rng-and-ordered-update-reference
  (let [W (seeded 6 31) x (seeded 6 32) tgt (seeded 4 33)]
    (doseq [[key values] [[:W W] [:x x] [:tgt tgt]]]
      (is (= (decoded-bits (get initial-bits key)) (bits values))))
    (is (= (Integer/parseUnsignedInt "3c23d70a" 16) (Float/floatToRawIntBits (float 0.01))))
    (loop [step 0 before (vec W)]
      (when (< step 2)
        (let [after (ordered-update before (vec x) (vec tgt))]
          (is (= (decoded-bits (nth ordered-update-bits step)) (bits after)))
          (is (not= (bits before) (bits after)))
          (recur (inc step) after))))))

(deftest portable-training-mathematical-gradient-is-an-independent-coordinate-derivative
  (let [x (mapv exact-float (seeded 6 32))
        tgt (mapv exact-float (seeded 4 33))
        states [(mapv exact-float (seeded 6 31))
                (mapv exact-float (decoded-floats (first ordered-update-bits)))]]
    (doseq [[state expected] (map vector states mathematical-reference)]
      (let [actual (coordinate-reference state x tgt)
            epsilon 1/1024]
        (is (= (:loss expected) (:loss actual)))
        (is (= (:dW expected) (:dW actual)))
        (doseq [i (range 6)]
          (let [upper (:loss (coordinate-reference (update state i + epsilon) x tgt))
                lower (:loss (coordinate-reference (update state i - epsilon) x tgt))]
            (is (= (nth (:dW actual) i) (/ (- upper lower) (* 2 epsilon)))
                "exact central difference of the quadratic primal checks each W coordinate")))))))

(deftest portable-dyadic-primal-and-weight-gradient-reference
  (doseq [{:keys [W loss dW]} dyadic-cases]
    (let [actual (coordinate-reference W dyadic-input dyadic-target)
          epsilon 1/1024]
      (is (= loss (:loss actual)))
      (is (= dW (:dW actual)))
      (doseq [i (range 6)]
        (is (= (nth dW i)
               (/ (- (:loss (coordinate-reference (update W i + epsilon) dyadic-input dyadic-target))
                     (:loss (coordinate-reference (update W i - epsilon) dyadic-input dyadic-target)))
                  (* 2 epsilon)))))))
  (is (not= (:loss (first dyadic-cases)) (:loss (second dyadic-cases))))
  (is (every? true? (map not= (:dW (first dyadic-cases)) (:dW (second dyadic-cases))))))

(defn- normalized-weight-error [actual expected]
  ;; Preserve the committed fixture's maximum error and denominator unchanged.
  (/ (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) actual expected))
     (reduce max 1e-9 (map #(Math/abs (double %)) expected))))

(defn- prove-resident-training! []
  (let [W (seeded 6 31) x (seeded 6 32) tgt (seeded 4 33)
        initial (vec W)
        prepared (compiled/lower #'tiny-training-step [W x tgt (long 2) (long 3) (long 2) 0.01]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :constants '[x tgt] :donate '[W]})
        plan (compiled/plan prepared)
        witness (invocation-link/memory-witness (invocation-link/certify plan))
        {:keys [compiler-buffer-bindings compiler-values semantic-outputs memory value-versions]} witness]
    (is (= :none (get-in prepared [:schedule :stats :fallback])))
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    (is (= :semantic-equation-read-before-write (:ordering value-versions)))
    (is (= :unproven (:completion value-versions)))
    (is (seq (:storage value-versions)))
    (is (some (fn [{:keys [retention definition public-output?]}]
                (and (= :equation (:kind definition)) (not public-output?)
                     (some #(> (:equation-index %)
                               (get-in retention [:definition-position :equation-index]))
                           (:use-positions retention))))
              (vals compiler-values))
        "internal forward/gradient values remain live through later AD equations")
    (is (seq compiler-buffer-bindings))
    (is (= (set (keys compiler-buffer-bindings)) (set (keys compiler-values))))
    (is (some #(= :equation (get-in % [:definition :kind])) (vals compiler-values)))
    (is (seq semantic-outputs))
    (is (every? #(contains? (:values memory) %) (vals compiler-buffer-bindings)))
    (is (every? #(contains? (:values memory) (second %)) semantic-outputs))
    (is (seq (:accesses memory)))
    (is (= :unproven (:reuse memory)))
    (let [program (compiled/instantiate! prepared)
          prior (volatile! nil)
          previous (volatile! initial)]
      (try
        (doseq [step (range 2)]
          (let [old @prior
                result (with-redefs [gpu-link/write!
                                     (fn [& _] (throw (ex-info "resident training replay uploaded an input" {})))]
                         (program (if old {:W old} {})))
                next-weights (:W' result)
                actual (value/->host next-weights)
                expected (decoded-floats (nth ordered-update-bits step))]
            (when old (is (not (value/live? old)) "donation consumes the previous W handle"))
            (is (value/live? next-weights))
            (vreset! prior next-weights)
            (is (= 6 (count actual) (count expected)))
            (is (< (normalized-weight-error actual expected) 1e-3)
                "resident updates match independent expected weights under the existing bound")
            (is (not= initial expected))
            (is (not= @previous (vec actual)) "each resident update advances W")
            (vreset! previous (vec actual))))
        (is (= initial (vec W)) "the host W initializer is not the evolving resident state")
        (finally (compiled/close! program)))
      (is (not (value/live? @prior)) "closing the owner invalidates its final W projection"))))

(deftest public-cuda-tiny-training-donates-resident-weights-without-replay-uploads
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "tiny training qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)) "required training execution cannot skip an absent device")
      (when (= :ready (:status device)) (prove-resident-training!)))))

(defn- prove-public-dyadic-weight-gradient! []
  (let [W (float-array (:W (first dyadic-cases)))
        x (float-array dyadic-input) tgt (float-array dyadic-target)
        prepared (compiled/lower #'tiny-training-weight-gradient
                                 [W x tgt (long 2) (long 3) (long 2)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :constants '[x tgt]})
        plan (compiled/plan prepared)
        output-contract (get-in plan [:nodes (first (:outputs plan)) :view])]
    ;; Public output metadata is inspected before reading numerical results.
    ;; No private AD intermediate allocation is downloaded or exposed.
    (is (= :none (get-in prepared [:schedule :stats :fallback])))
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    (is (= :float (:dtype output-contract)))
    (is (= [6] (:shape output-contract)))
    (println "[CUDA AD WEIGHT GRADIENT OUTPUT CONTRACT]"
             (select-keys output-contract [:dtype :shape :byte-length]))
    (let [program (compiled/instantiate! prepared)
          prior (volatile! nil)]
      (try
        (doseq [[index {:keys [W dW]}] (map-indexed vector dyadic-cases)]
          (let [old @prior
                outputs (program (if (zero? index) {} {:W (float-array W)}))
                output (first (vals outputs))]
            ;; This separate non-donated numerical probe deliberately updates
            ;; its public W input. It does not claim no-upload resident training.
            (is (= 1 (count outputs)))
            (is (value/device-array? output))
            (is (= :float (:dtype output)))
            (is (= [6] (:shape output)))
            (when old (is (not (value/live? old))))
            (is (value/live? output))
            (vreset! prior output)
            (is (= (bits dW) (bits (value/->host output)))
                "actual AD-returned dW matches exact independent dyadic coordinates")))
        (is (= (bits (:W (first dyadic-cases))) (bits W)))
        (finally (compiled/close! program)))
      (is (not (value/live? @prior))))))

(deftest public-cuda-ad-weight-gradient-matches-two-exact-dyadic-cases
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "direct AD gradient qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-public-dyadic-weight-gradient!)))))

(defn- prove-public-dyadic-primal! []
  (let [W (float-array (:W (first dyadic-cases)))
        initial (vec W)
        prepared (compiled/lower #'tiny-training-loss
                                 [W (float-array dyadic-input) (float-array dyadic-target)
                                  (long 2) (long 3) (long 2)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :constants '[x tgt]})
        plan (compiled/plan prepared)
        view (get-in plan [:nodes (first (:outputs plan)) :view])
        program (compiled/instantiate! prepared)
        prior (volatile! nil)]
    (try
      (is (= :none (get-in prepared [:schedule :stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations])))
      (is (= :double (:dtype view)))
      (is (= [] (:shape view)))
      (is (= 8 (:byte-length view)))
      (println "[CUDA PRIMAL LOSS OUTPUT CONTRACT]"
               (select-keys view [:dtype :shape :byte-length]))
      (doseq [[index {:keys [W loss]}] (map-indexed vector dyadic-cases)]
        (let [old @prior
              outputs (program (if (zero? index) {} {:W (float-array W)}))
              output (first (vals outputs))
              host (value/->host output)]
          ;; This non-donated primal probe deliberately uploads changed W.
          ;; The separate resident SGD case retains its no-upload replay contract.
          (is (= 1 (count outputs)))
          (is (value/device-array? output))
          (is (= :double (:dtype output)))
          (is (= [] (:shape output)))
          (is (= 1 (count host)))
          (is (value/live? output))
          (when old (is (not (value/live? old))))
          (vreset! prior output)
          (is (= (Double/doubleToRawLongBits (double loss))
                 (Double/doubleToRawLongBits (first host)))
              "actual declared Double loss matches the exact independent dyadic reference")))
      (is (= initial (vec W)))
      (finally (compiled/close! program)))
    (is (not (value/live? @prior)))))

(defn- prove-public-double-division! []
  ;; The same unchanged MSE reduction/division, fresh independent execution.
  ;; Exact Double 1/3 differs from Float division followed by widening.
  (let [prepared (compiled/lower #'tiny-training-mse-primal
                                 [(float-array [1 0 0]) (float-array [0 0 0]) (long 3)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :constants '[pred target]})
        program (compiled/instantiate! prepared)
        result (volatile! nil)]
    (try
      (let [outputs (program {})
            output (first (vals outputs))
            host (value/->host output)]
        (vreset! result output)
        (is (= 1 (count outputs)))
        (is (= :double (:dtype output)))
        (is (= [] (:shape output)))
        (is (= 1 (count host)))
        (is (= "3fd5555555555555"
               (format "%016x" (Double/doubleToRawLongBits (first host)))))
        (is (not= "3fd5555560000000"
                  (format "%016x" (Double/doubleToRawLongBits (first host))))))
      (finally (compiled/close! program)))
    (is (not (value/live? @result)))))

(deftest public-cuda-primal-loss-matches-two-exact-dyadic-cases
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "primal loss qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-public-dyadic-primal!)))))

(deftest public-cuda-mse-retains-double-division-precision
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "Double division qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-public-double-division!)))))
