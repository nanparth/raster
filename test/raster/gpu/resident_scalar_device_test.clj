(ns raster.gpu.resident-scalar-device-test
  "Retained sequential scalar regions through the public Compiled/LinkPlan route."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.passes.parallel.scheduled-equation-graph-test :as schedules]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.nn :as nn]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-runtime :as cuda]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.ocl-runtime :as runtime]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.value :as value]))

(deftest public-lowering-certifies-scalar-and-array-results
  ;; This is allocation-free compilation against a recorded descriptor, not a native claim.
  (with-redefs [hardware/descriptor-for
                (fn [_] @#'schedules/scalar-source-target)]
    (doseq [[source arguments dtype shape bytes]
            [[#'array-ops/masked-mse-loss
              [(double-array [1 2 3 4]) (double-array [1.5 2 2.5 3.5])
               (long-array [0 1 0 0]) (long 4)] :double [] 8]
             [#'nn/softmax-1d-backward
              [(float-array [1 2 3 4]) (float-array [0.25 0.25 0.25 0.25])
               (long 4)] :float [4] 16]]]
      (let [prepared (compiled/lower source arguments
                                     {:compiler :equation-first :target :ocl:0
                                      :dtype dtype :on-non-resident :throw})
            outputs (:out-tree prepared)
            output (first outputs)
            node (get-in (compiled/plan prepared) [:nodes (:node output)])]
        (is (compiled/prepared? prepared))
        (is (= 1 (count outputs)))
        (is (= :result (:key output)) "the unchanged semantic return is the public result")
        (is (= dtype (:dtype output)))
        (is (= shape (:shape output)))
        (is (= shape (get-in node [:view :shape])))
        (is (= bytes (get-in node [:view :byte-length])))))))

(defn- raw-bits [dtype values]
  (mapv (case dtype
          :float #(Float/floatToRawIntBits %)
          :double #(Double/doubleToRawLongBits %)
          :long identity) values))

(defn- prove-replays! [target runtime-state source arguments dtype shape initial-inputs replays]
  (let [baseline (root/lease-count runtime-state)
        prepared (compiled/lower source arguments
                                 {:compiler :equation-first :target target
                                  :dtype dtype :on-non-resident :throw})
        output-entry (first (:out-tree prepared))
        program (compiled/instantiate! prepared)
        prior (atom nil)
        input-state (atom initial-inputs)]
    (try
      (is (= :result (:key output-entry)))
      (doseq [{:keys [inputs expected]} replays]
        ;; Public Compiled refreshes omitted dynamic inputs from its captured defaults on
        ;; every call. Keep the partial third replay to check that restoration explicitly.
        (reset! input-state (merge initial-inputs inputs))
        (let [output (:result (program inputs))
              actual (value/->host output)]
          (when @prior
            (is (not (value/live? @prior)) "replay retires the previous public output"))
          (reset! prior output)
          (is (value/device-array? output))
          (is (= dtype (:dtype output)))
          (is (= shape (:shape output)))
          (is (= (count expected) (alength actual)) "cardinality precedes exact bit comparison")
          (is (= (raw-bits dtype expected) (raw-bits dtype actual))
              "independent exact dyadic reference, including positive-zero bits")
          (let [session (get-in program [:executable :session])
                physical (get-in program [:executable :node-views (:node output-entry)])]
            (is (= shape (get-in physical [:view :shape])))
            (is (= (get-in (compiled/plan prepared) [:nodes (:node output-entry) :view :byte-length])
                   (get-in physical [:view :byte-length])))
            (is (empty? (:events @session)) "synchronous public replay leaves no completion debt")
            (doseq [[key expected-input] @input-state
                    :let [entry (first (filter #(= key (:key %)) (:in-tree prepared)))
                          input-view (get-in program [:executable :node-views (:node entry)])
                          observed (aclone expected-input)]]
              (is (some? entry))
              (gpu/download-range! session input-view observed {:elements (alength expected-input)})
              (is (= (raw-bits (:dtype entry) expected-input)
                     (raw-bits (:dtype entry) observed))
                  "all input elements, including unused padding guards, remain unchanged")))))
      (println "[PUBLIC RETAINED SCALAR COMPLETED]"
               {:source (str source) :target target :dtype dtype :shape shape :replays (count replays)
                :device (gpu/execution-device-info (get-in program [:executable :session]))
                :execution (compiled/execution-info program)})
      (finally (compiled/close! program)))
    (is (not (value/live? @prior)) "closing the program invalidates its last output")
    (is (= baseline (root/lease-count runtime-state))
        "session, prepared calls and buffers retire to the shared runtime baseline")))

(defn- prove-scalar-cases! [target runtime-state]
  ;; Independently checked Fraction witnesses: every intermediate in these small cases
  ;; is exactly representable; this does not assert universal Float fold bit parity.
  (prove-replays!
   target runtime-state #'array-ops/masked-mse-loss
   [(double-array [1 2 3 4 137 139]) (double-array [1.5 2 2.5 3.5 149 151])
    (long-array [0 1 0 0 157 163]) (long 4)] :double []
   {:pred (double-array [1 2 3 4 137 139])
    :target (double-array [1.5 2 2.5 3.5 149 151])
    :states (long-array [0 1 0 0 157 163])}
   [{:inputs {} :expected [0.25]}
    {:inputs {:pred (double-array [2 4 6 8 137 139]) :target (double-array [1 2 4 5 149 151])
              :states (long-array [1 0 1 0 157 163])} :expected [6.5]}
    {:inputs {:states (long-array [1 1 1 1 157 163])} :expected [0.0]}])
  (prove-replays!
   target runtime-state #'nn/softmax-1d-backward
   [(float-array [1 2 3 4 137 139]) (float-array [0.25 0.25 0.25 0.25 149 151]) (long 4)]
   :float [4]
   {:dy (float-array [1 2 3 4 137 139])
    :result (float-array [0.25 0.25 0.25 0.25 149 151])}
   [{:inputs {} :expected [-0.375 -0.125 0.125 0.375]}
    {:inputs {:dy (float-array [3 -1 2 -4 137 139])
              :result (float-array [0.5 0.25 0.125 0.125 149 151])}
     :expected [1.0 -0.5 0.125 -0.625]}
    {:inputs {:dy (float-array [0 0 0 0 137 139])} :expected [0.0 0.0 0.0 0.0]}]))

(deftest public-retained-scalar-and-ordered-backward-replay-on-opencl
  (let [available @probe/opencl-fp64-available?]
    (when (= "1" (System/getenv "RASTER_REQUIRE_RESIDENT_SCALAR_DEVICE"))
      (is available "required scalar native qualification cannot skip the FP64 device"))
    (if available
      (prove-scalar-cases! :ocl:0 @#'runtime/state)
      (probe/opencl-skip! "public retained sequential scalar runtime" :fp64))))

(deftest public-retained-scalar-and-ordered-backward-replay-on-cuda
  (if (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (if @probe/cuda-available?
      (do
        (cuda/ensure-init!)
        (prove-scalar-cases! :cuda:0 cuda/state))
      (probe/cuda-skip! "public retained sequential scalar CUDA runtime"))
    (probe/cuda-skip! "public retained sequential scalar CUDA runtime" :not-requested)))
