(ns raster.gpu.resident-device-admission-test
  "Opt-in native qualification of the public CUDA Compiled/LinkPlan path."
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver-probe :as cuda-probe]
            [raster.gpu.value :as value]
            [raster.par :as par]))

(deftm affine!
  [input :- (Array float) result :- (Array float) n :- Long] :- Void
  (par/map-void! i n
    (arrays/aset result i (+ (* 2.0 (arrays/aget input i)) 1.0))))

(deftest public-cuda-compiled-map-replays-and-releases-output-values
  (if-not (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
    (println "SKIP public CUDA admission execution; set RASTER_REQUIRE_CUDA_DEVICE=1")
    (let [probe (cuda-probe/probe)]
      (is (= :ready (:status probe)) "required CUDA device discovery must succeed")
      (when (= :ready (:status probe))
        (let [n 259
              initial (float-array (map #(float (/ % 8.0)) (range n)))
              changed (float-array (map #(float (/ (- % 71) 4.0)) (range n)))
              reference (fn [input] (mapv #(float (+ (* 2.0 %) 1.0)) input))
              prepared (compiled/lower #'affine! [initial (float-array n) (long n)]
                                       {:compiler :equation-first :target :cuda:0 :dtype :float
                                        :outputs '[result] :on-non-resident :throw})
              _ (is (compiled/prepared? prepared))
              _ (is (zero? (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
              program (compiled/instantiate! prepared)
              last-output (volatile! nil)]
          (try
            (let [identity (gpu/execution-device-info (get-in program [:executable :session]))]
              (is (= 0 (:device-ordinal identity)))
              (println :PUBLIC-CUDA-EXECUTION {:elements n :dtype :float :device identity}))
            (let [first-output (:result (program {}))]
              (vreset! last-output first-output)
              (is (= (reference initial) (vec (value/->host first-output))))
              (let [next-output (:result (program {:input changed}))]
                (vreset! last-output next-output)
                (is (not (value/live? first-output)) "replay invalidates the prior output projection")
                (is (= (reference changed) (vec (value/->host next-output))))))
            (is (seq (compiled/execution-info program)) "the public artifact bound executable kernels")
            (finally (compiled/close! program)))
          (is (not (value/live? @last-output)) "closing the owner invalidates output projections"))))))
