(ns raster.gpu.cuda-f32-arithmetic-device-test
  "Retained float multiply/add and explicit FMA through public Compiled/LinkPlan."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.value :as value]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.math :as math]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.runtime.hardware :as hardware]))

(deftm arithmetic!
  [a :- (Array float) b :- (Array float) c :- (Array float)
   separate :- (Array float) fused :- (Array float) n :- Long] :- Void
  (par/map-void! i n
    (do
      (arrays/aset separate i
                   (numeric/+ (numeric/* (arrays/aget a i) (arrays/aget b i))
                              (arrays/aget c i)))
      (arrays/aset fused i
                   (math/fma (arrays/aget a i) (arrays/aget b i) (arrays/aget c i))))))

(defn- float-bits [values]
  (mapv #(Float/floatToRawIntBits (float %)) values))

(defn- triples []
  (let [a (float (+ 1.0 (Math/scalb 1.0 (int -13))))
        b (float (- 1.0 (Math/scalb 1.0 (int -13))))]
    [[a b -1.0] [b a -1.0] [(- a) b 1.0] [1.0 1.0 1.0]]))

(defn- inputs [n shift]
  (let [cases (triples)]
    (into {} (map-indexed
              (fn [column key]
                [key (float-array (map #(nth (nth cases (mod (+ % shift) 4)) column)
                                      (range n)))]) [:a :b :c]))))

(defn- independent-bits [{:keys [a b c]} n]
  ;; These dyadic products/sums are exact in binary64. This is not a generic FMA oracle.
  (let [rows (map vector (take n a) (take n b) (take n c))]
    {:separate (float-bits (map (fn [[x y z]]
                                 (float (+ (double (float (* (double x) (double y))))
                                           (double z)))) rows))
     :fused (float-bits (map (fn [[x y z]]
                              (float (+ (* (double x) (double y)) (double z)))) rows))}))

(defn- lower [n]
  (let [{:keys [a b c]} (inputs n 0)
        guarded (fn [] (float-array (concat (repeat n 123.0) [456.0 -789.0])))]
    (compiled/lower #'arithmetic! [a b c (guarded) (guarded) (long n)]
                    {:compiler :equation-first :target :cuda:0 :dtype :float
                     :schedule {:precision :f32-scalar}
                     :outputs '[separate fused]
                     :roles {'separate :state 'fused :state}
                     :on-non-resident :throw})))

(deftest dyadic-host-oracle-distinguishes-retained-and-fused-rounding
  (let [bits (independent-bits (inputs 4 0) 4)]
    (is (= [0 0 0 1073741824] (:separate bits)))
    (is (= [(unchecked-int 0xb2800000) (unchecked-int 0xb2800000)
            847249408 1073741824] (:fused bits)))))

(deftest public-map-lowering-remains-device-free
  (hardware-fixture/isolated
   (fn []
     (hardware/register-target-device! :cuda:0 {:name "NVIDIA A100"})
     (let [prepared (lower 257)]
       (is (compiled/prepared? prepared))
       (is (zero? (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
       (is (= :state (get-in (compiled/plan prepared) [:attributes :public-buffer-roles 'separate])))
       (is (= :state (get-in (compiled/plan prepared) [:attributes :public-buffer-roles 'fused])))))))

(defn- prove-outputs! [program outputs expected n]
  (let [session (get-in program [:executable :session])]
    (doseq [key [:separate :fused]]
      (let [actual (get outputs key)
            buffer-key (first (for [[key buffer] (:buffers @session)
                                    :when (identical? (cleanup/lifetime-owner buffer)
                                                      (cleanup/lifetime-owner (:buffer actual)))] key))
            buffer (gpu/buffer session buffer-key)
            view (gpu/buffer-view session buffer-key {:shape [(+ n 2)]})
            physical (float-array (+ n 2))]
        (is (= (get expected key) (float-bits (take n (value/->host actual))))
            (str key " exact independent float bits"))
        (is (= (* 4 (+ n 2)) (:byte-size buffer)) "physical state includes trailing guards")
        (gpu/download-range! session view physical {:elements (+ n 2)})
        (is (= (get expected key) (float-bits (take n physical))))
        (is (= (float-bits [456.0 -789.0]) (float-bits (drop n physical)))
            (str key " masked tail preserves initialized guard bits"))))))

(defn- offset-inputs! [program n]
  (let [session (get-in program [:executable :session])
        changed (inputs n 2)]
    (into {} (for [[key host] changed]
               (let [buffer-key (keyword "f32-view" (name key))
                     source (float-array (concat [999.0] host [-999.0]))]
                 (gpu/alloc! session {buffer-key [:float (+ n 2) source]})
                 (let [view (gpu/buffer-view session buffer-key {:byte-offset 4 :shape [n]})]
                   [key (value/wrap-external-view (gpu/buffer session buffer-key)
                                                  :cuda:0 (:view view))]))))))

(defn- save-artifacts! [program]
  (when-let [directory (System/getenv "RASTER_F32_EVIDENCE_DIR")]
    (let [directory (io/file directory)]
      (.mkdirs directory)
      (spit (io/file directory "execution.edn") (pr-str (compiled/execution-info program)))
      (spit (io/file directory "device.edn") (pr-str (runtime/execution-device-info)))
      (doseq [[index [_ artifact]] (map-indexed vector @(var-get #'runtime/kernel-registry))]
        (spit (io/file directory (str "kernel-" index ".cu")) (:source artifact))
        (when-let [compiled @(get artifact :raster.gpu.cuda-runtime/compiled)]
          (with-open [out (io/output-stream (io/file directory (str "kernel-" index ".ptx")))]
            (.write out (:ptx compiled)))
          (spit (io/file directory (str "kernel-" index "-identity.edn"))
                (pr-str (dissoc compiled :ptx))))))))

(deftest public-cuda-f32-map-preserves-retained-operations-and-explicit-fma
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (device-probe/cuda-skip! "f32 arithmetic requires RASTER_REQUIRE_CUDA_DEVICE=1" :not-requested)
    (let [device (probe/probe)]
      (is (= :ready (:status device)) "required native arithmetic cannot skip")
      (when (= :ready (:status device))
        (let [n 257 program (compiled/instantiate! (lower n))
              last-output (atom nil) views (atom {})]
          (try
            (let [first-output (program {})]
              (reset! last-output first-output)
              (prove-outputs! program first-output (independent-bits (inputs n 0) n) n)
              (let [second-output (program (inputs n 1))]
                (is (every? #(not (value/live? %)) (vals first-output)))
                (reset! last-output second-output)
                (prove-outputs! program second-output (independent-bits (inputs n 1) n) n)
                (reset! views (offset-inputs! program n))
                (let [third-output (program @views)]
                  (is (every? #(not (value/live? %)) (vals second-output)))
                  (reset! last-output third-output)
                  (prove-outputs! program third-output (independent-bits (inputs n 2) n) n))))
            (save-artifacts! program)
            (finally
              (doseq [view (vals @views)] (value/free! view))
              (compiled/close! program)))
          (is (every? #(not (value/live? %)) (vals @last-output))))))))
