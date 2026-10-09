(ns raster.gpu.cuda-matrix-device-test
  "Exact FP32 NN accumulating GEMM on public equation-first Compiled/LinkPlan."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.gemm-epilogue-device-test :as gemm]
            [raster.core :refer [deftm]]
            [raster.linalg.blas :as blas]
            [raster.numeric :as numeric]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.value :as value]))

;; Portable copies of the tiny rational fixtures prepared on 2026-10-08:
;; evidence/matrix-reference/fixtures.json SHA256
;; 9ab34afd937bd3f1cbd96a4d310866a893b2bad12d494a90d829d5ea4889847c.
;; No workbench evidence is loaded at runtime. All products/partial sums below
;; are exactly representable binary32 dyadics; this is not a general GEMM oracle.
;; Existing accumulate-gemm! uses NN A[i*4+l], B[l*5+j], alpha=1, beta=1.
(def expected-reset
  [[11.375 3.75 -3.875 -11.5 -13.375 -5.5 -2.5 0.5 3.5 15.125 -5.75 -4.0 -2.25 -0.5 12.75]
   [-1.5 -1.625 9.75 -16.25 0.875 1.25 -0.125 -15.875 14.375 4.375 4.0 1.375 13.125 -9.625 7.875]
   [-10.0 11.375 4.0 -0.5 -5.0 7.625 -12.125 -6.0 0.125 6.25 -8.0 16.625 12.5 5.5 -1.5]])

(def expected-carried
  [[11.375 3.75 -3.875 -11.5 -13.375 -5.5 -2.5 0.5 3.5 15.125 -5.75 -4.0 -2.25 -0.5 12.75]
   [13.125 4.875 8.125 -26.0 -11.25 -3.5 -2.375 -15.625 17.125 18.25 -3.5 -4.875 8.125 -13.375 16.875]
   [6.375 19.0 14.375 -24.75 -15.0 4.875 -14.25 -21.875 16.5 23.25 -13.25 9.5 17.875 -11.125 11.625]])

(def expected-nt-reset
  [[6.5 -14.0 -14.375 -9.0 10.75 -4.375 -5.375 13.75 12.75 -2.625 -12.875 12.75 3.875 3.625 14.875]
   [0.375 1.25 2.125 -14.25 -13.375 2.0 3.625 5.25 15.5 17.125 3.625 6.0 8.375 -9.375 -7.0]
   [10.25 -9.25 -5.75 -5.125 -1.625 -10.75 -10.75 -13.625 6.5 3.625 15.75 -5.125 -0.125 -0.875 4.125]])

;; The existing FP32 dgemm-nt! overload stores B as [n,k], alpha=beta=1.
;; This is a separate registered BLAS operation, not an NN transpose flag.
(deftm accumulate-gemm-nt!
  [A :- (Array float) B :- (Array float) C :- (Array float)
   m :- Long k :- Long n :- Long] :- (Array float)
  (let [_ (blas/dgemm-nt! A B C m k n (numeric/oftype A 1.0) (numeric/oftype A 1.0))]
    C))

(defn- bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- inputs [step]
  {:A (float-array (map #(float (/ (- (mod (+ (* 7 %) (* 3 step)) 19) 9) 4.0)) (range 12)))
   :B (float-array (map #(float (/ (- (mod (+ (* 5 %) (* 7 step)) 23) 11) 2.0)) (range 20)))})

(defn- initial-c []
  (mapv #(float (/ (- (* 2 %) 13) 4.0)) (range 15)))

(defn- independent-nn [{:keys [A B]} before]
  ;; Scalar host coordinate loop, without Raster/BLAS numerical operations.
  (vec (for [i (range 3) j (range 5)]
         (float (+ (double (nth before (+ (* i 5) j)))
                   (reduce + 0.0
                           (for [l (range 4)]
                             (* (double (aget A (+ (* i 4) l)))
                                (double (aget B (+ (* l 5) j)))))))))))

(deftest portable-matrix-reference-distinguishes-carried-and-reset-state
  (loop [step 0 before (initial-c)]
    (when (< step 3)
      (let [current (inputs step)
            carried (independent-nn current before)
            reset (independent-nn current (initial-c))]
        (is (= (bits (nth expected-carried step)) (bits carried)))
        (is (= (bits (nth expected-reset step)) (bits reset)))
        (when (pos? step)
          (is (not= (bits carried) (bits reset)))
          (doseq [key [:A :B]]
            (is (not= (bits (get (inputs (dec step)) key)) (bits (get current key))))))
        (recur (inc step) carried)))))

(deftest portable-nt-reference-distinguishes-the-existing-b-layout
  (doseq [step (range 3)]
    (let [{:keys [A B]} (inputs step)
          before (initial-c)
          nt (for [i (range 3) j (range 5)]
               (float (+ (double (nth before (+ (* i 5) j)))
                         (reduce + 0.0
                                 (for [l (range 4)]
                                   (* (double (aget A (+ (* i 4) l)))
                                      (double (aget B (+ (* j 4) l)))))))))]
      (is (= (bits (nth expected-nt-reset step)) (bits nt)))
      (is (not= (bits (nth expected-reset step)) (bits nt))))))

(defn- physical-result! [program output]
  (let [session (get-in program [:executable :session])
        key (first (for [[key buffer] (:buffers @session)
                         :when (identical? (cleanup/lifetime-owner buffer)
                                           (cleanup/lifetime-owner (:buffer output)))] key))
        buffer (gpu/buffer session key)
        view (gpu/buffer-view session key {:shape [18]})
        host (float-array 18)]
    (is (= 72 (:byte-size buffer)) "owned C retains its fifteen elements and three guards")
    (gpu/download-range! session view host {:elements 18})
    host))

(defn- resident-inputs! [program current]
  (let [session (get-in program [:executable :session])]
    (into {} (for [[key host] current]
               (let [offset (if (= key :A) 3 5)
                     owner (float-array (concat (repeat offset 4096.0) host (repeat 3 -4096.0)))
                     allocation-key (keyword "matrix-input" (name key))]
                 (gpu/alloc! session {allocation-key [:float (alength owner) owner]})
                 (let [view (gpu/buffer-view session allocation-key
                                             {:byte-offset (* 4 offset) :shape [(alength host)]})]
                   [key {:value (value/wrap-external-view (gpu/buffer session allocation-key)
                                                          :cuda:0 (:view view))
                         :allocation-key allocation-key :owner-bits (bits owner)}]))))))

(defn- prove-carried-state! []
  (let [{:keys [A B]} (inputs 0)
        C (float-array (concat (initial-c) (repeat 3 -4096.0)))
        ;; C is initialized once and captured as owned state at offset zero.
        ;; No :C invocation keyword or sealed-plan rewrite resets that state.
        prepared (compiled/lower #'gemm/accumulate-gemm! [A B C (long 3) (long 4) (long 5)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :schedule {:precision :f32-scalar}
                                  :outputs '[C] :roles {'C :state} :on-non-resident :throw})
        program (compiled/instantiate! prepared)
        last-output (atom nil) resident (atom {})]
    (try
      (is (compiled/prepared? prepared))
      (is (= :state (get-in (compiled/plan prepared) [:attributes :public-buffer-roles 'C])))
      (doseq [step (range 3)]
        (let [current (inputs step)
              supplied (if (= step 2)
                         (do (reset! resident (resident-inputs! program current))
                             (into {} (map (fn [[key entry]] [key (:value entry)]) @resident)))
                         current)
              previous @last-output
              output (:C (program supplied))
              expected (nth expected-carried step)
              physical (physical-result! program output)]
          (reset! last-output output)
          (when previous (is (not (value/live? previous))))
          (is (= (bits expected) (bits (take 15 (value/->host output)))))
          (is (= (bits expected) (bits (take 15 physical))))
          (is (= (bits (repeat 3 -4096.0)) (bits (drop 15 physical)))
              "three C guards remain unchanged across carried-state replays")))
      ;; Dynamic DeviceArray inputs are copied into the public program's owned
      ;; inputs. This is offset-view D2D coverage, not direct kernel-pointer proof.
      (doseq [[_ {:keys [allocation-key owner-bits]}] @resident]
        (is (= owner-bits (bits (gpu/download (get-in program [:executable :session]) allocation-key)))
            "input owner prefix/suffix and data are unchanged"))
      (finally
        (doseq [[_ entry] @resident] (value/free! (:value entry)))
        (compiled/close! program)))
    (is (not (value/live? @last-output)))))

(deftest public-cuda-nn-gemm-carries-initial-c-through-two-changed-input-replays
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "matrix qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)) "required CUDA matrix execution cannot skip")
      (when (= :ready (:status device)) (prove-carried-state!)))))

(defn- prove-nt-reset! []
  ;; Fresh owned state per case explicitly resets C. These are three independent
  ;; executions with changed A/B, not carried-state NT replay.
  (doseq [step (range 3)]
    (let [{:keys [A B]} (inputs step)
          C (float-array (concat (initial-c) (repeat 3 -4096.0)))
          prepared (compiled/lower #'accumulate-gemm-nt! [A B C (long 3) (long 4) (long 5)]
                                   {:compiler :equation-first :target :cuda:0 :dtype :float
                                    :schedule {:precision :f32-scalar}
                                    :outputs '[C] :roles {'C :state} :on-non-resident :throw})
          program (compiled/instantiate! prepared)
          last-output (atom nil)]
      (try
        (let [output (:C (program {}))
              physical (physical-result! program output)
              public-output (value/->host output)]
          (reset! last-output output)
          ;; The equation returns the full caller C allocation, including its tail.
          (is (= 18 (alength public-output)))
          (is (= (bits (concat (nth expected-nt-reset step) (repeat 3 -4096.0)))
                 (bits public-output)))
          (is (= (bits (nth expected-nt-reset step)) (bits (take 15 physical))))
          (is (= (bits (repeat 3 -4096.0)) (bits (drop 15 physical)))))
        (finally (compiled/close! program)))
      (is (not (value/live? @last-output))))))

(deftest public-cuda-nt-gemm-uses-separate-reset-state-executions
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "NT qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-nt-reset!)))))

(defn- padded-owner [offset values]
  (float-array (concat (repeat offset 4096.0) values (repeat 3 -4096.0))))

(defn- prove-direct-nn-offsets! []
  (let [{:keys [A B]} (inputs 0)
        prepared (compiled/lower #'gemm/accumulate-gemm!
                                 [A B (float-array (concat (initial-c) (repeat 3 -4096.0)))
                                  (long 3) (long 4) (long 5)]
                                 {:compiler :equation-first :target :cuda:0 :dtype :float
                                  :schedule {:precision :f32-scalar}
                                  :outputs '[C] :roles {'C :state} :on-non-resident :throw})
        step (get-in (compiled/plan prepared) [:instances 0 :call :steps 0])
        graph (:graph step)
        scalars (select-keys (:scalar-values step) (map :id (:scalars graph)))
        session (gpu/make-session :cuda:0)]
    (try
      (gpu/alloc! session {:A [:float 18 nil] :B [:float 28 nil] :C [:float 20 nil]})
      (let [a-view (gpu/buffer-view session :A {:byte-offset 12 :shape [12]})
            b-view (gpu/buffer-view session :B {:byte-offset 20 :shape [20]})
            c-view (gpu/buffer-view session :C {:byte-offset 8 :shape [15]})
            bound (gpu/bind-kernel-graph! session :matrix-offsets graph
                                         {'A a-view 'B b-view 'C c-view} scalars)]
        (is (= [12 20 8] (mapv #(get-in % [:view :byte-offset]) [a-view b-view c-view])))
        (doseq [pass (range 3)]
          (let [{:keys [A B]} (inputs pass)
                a-owner (padded-owner 3 A)
                b-owner (padded-owner 5 B)
                c-owner (padded-owner 2 (initial-c))]
            ;; Explicit full-owner upload resets C for this direct-graph pass.
            (gpu/upload! session :A a-owner)
            (gpu/upload! session :B b-owner)
            (gpu/upload! session :C c-owner)
            (with-redefs [value/->host (fn [& _] (throw (ex-info "matrix graph materialized a host value" {})))]
              (gpu/run-kernel-graph! session bound))
            (is (= (bits a-owner) (bits (gpu/download session :A))))
            (is (= (bits b-owner) (bits (gpu/download session :B))))
            (is (= (bits (padded-owner 2 (nth expected-reset pass)))
                   (bits (gpu/download session :C)))
                "actual C offset writes only its dense15 values, preserving every owner guard")))
        (println "[CUDA GENERATED GRAPH NN MATRIX OFFSETS]"
                 {:byte-offsets [12 20 8] :shape [3 4 5] :passes 3 :state :explicit-reset
                  :route :equation-first-generated-graph-public-runtime-binding}))
      (finally (gpu/close-session! session)))))

(deftest generated-nn-equation-graph-binds-actual-input-and-state-offsets
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "matrix pointer-offset qualification requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)))
      (when (= :ready (:status device)) (prove-direct-nn-offsets!)))))
