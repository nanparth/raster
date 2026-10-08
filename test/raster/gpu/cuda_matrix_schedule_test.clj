(ns raster.gpu.cuda-matrix-schedule-test
  "Host admission discriminator for the observed-limit NN matrix contract."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.gemm-epilogue-device-test :as gemm]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.cuda-matrix-device-test :as matrix]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.runtime.hardware :as hardware]))

(deftest observed-occupancy-width-must-produce-a-legal-contraction-schedule
  (hardware-fixture/isolated
   (fn []
     ;; Synthetic compiler facts, not a probe or occupancy/performance assertion.
     (hardware/register-target-device!
      :cuda:0 {:name "synthetic observed-limit matrix fixture"
               :capabilities {:compute-capability [8 6] :sm-count 40 :warp-size 32
                              :max-threads-per-block 1024 :max-warps-per-sm 48
                              :max-blocks-per-sm 16 :shared-memory-per-block 49152}})
     (let [{:keys [A B]} (#'matrix/inputs 0)
           C (float-array (concat (#'matrix/initial-c) (repeat 3 -4096.0)))
           construction (atom nil)
           constructor reduction/schedule
           result (try
                    {:prepared
                     (with-redefs [reduction/schedule
                                   (fn [options]
                                     (if (= :hardware-contraction-candidates (:strategy options))
                                       (try
                                         (let [scheduled (constructor options)]
                                           (reset! construction {:schedule scheduled})
                                           scheduled)
                                         (catch clojure.lang.ExceptionInfo error
                                           (reset! construction {:error (ex-data error)})
                                           (throw error)))
                                       (constructor options)))]
                       (compiled/lower #'gemm/accumulate-gemm!
                                       [A B C (long 3) (long 4) (long 5)]
                                       {:compiler :equation-first :target :cuda:0 :dtype :float
                                        :schedule {:precision :f32-scalar}
                                        :outputs '[C] :roles {'C :state}
                                        :on-non-resident :throw}))}
                    (catch clojure.lang.ExceptionInfo error {:error (ex-data error)}))]
       (is (some? @construction) "the public compiler reaches contraction schedule construction")
       (is (nil? (:error @construction))
           "legal observed thread limits must admit a valid power-of-two contraction schedule")
       (when-let [scheduled (:schedule @construction)]
         (is (= scheduled (reduction/validate-schedule! scheduled)))
         (is (= [:matrix :register-tiled :portable] (get-in scheduled [:tuning-space :families])))
         (is (= 512 (:workgroup-size scheduled)))
         (is (= [32 64 128 256 512] (get-in scheduled [:tuning-space :workgroup-size])))
         (is (every? #(and (pos? %) (zero? (bit-and % (dec %)))
                           (<= % (:workgroup-size scheduled)))
                     (get-in scheduled [:tuning-space :workgroup-size])))
         ;; A later admission failure is a separate native/LinkPlan gate, not a
         ;; reason to obscure the precise schedule-construction result here.
         (println "[MATRIX AFTER SCHEDULE]" (select-keys result [:error])))))))

(deftest candidate-envelope-retains-capacity-and-recomputes-geometry
  ;; Helper qualification is separate from the public compiler test above.
  ;; No tiny K cap: rank-positive candidate families retain their target capacity.
  (let [descriptor {:max-workgroup-size 1024 :machine-lanes 61440
                    :cache {:slm 49152}}
        candidate #'lower/contraction-candidate-grid
        grid (candidate :cuda:0 (segop/->KernelGrid nil 768 3072) 1025 :float descriptor)
        capped (:num-blocks grid)]
    (is (= 512 (:block-size grid)))
    (is (= 2048 (:shared-mem-bytes grid)))
    (is (= 1 (first (:values capped))))
    (let [[cap groups] (:values (second (:values capped)))]
      (is (= 480 cap))
      (is (= 1025 (:value groups)))
      (is (= 512 (:divisor groups))))
    (let [prior (candidate :cuda:0 (segop/->KernelGrid nil 256 1024) 4 :float descriptor)]
      (is (= 256 (:block-size prior)))
      (is (= 1024 (:shared-mem-bytes prior))))
    (let [limited (candidate :cuda:0 (segop/->KernelGrid nil 768 3072) 4 :float
                             (assoc descriptor :max-workgroup-size 128 :cache {:slm 256}))]
      (is (= 64 (:block-size limited)))
      (is (= 256 (:shared-mem-bytes limited))))
    (doseq [invalid [(assoc descriptor :max-workgroup-size 0)
                     (assoc descriptor :cache {:slm 3})]]
      (is (= :contraction-candidate-target-limits
             (try (candidate :cuda:0 (segop/->KernelGrid nil 768 3072) 4 :float invalid)
                  nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= :reduction-schedule-workgroup
           (try (reduction/schedule {:strategy :hardware-contraction-candidates
                                     :workgroup-size 768})
                nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
