(ns raster.gpu.cuda-runtime-device-test
  "Required-device proof of existing equation-first generated CUDA, never a handwritten replacement."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-compile-fixtures :as fixtures]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.runtime.hardware :as hardware]))

(defn- device-present? [] @device-probe/cuda-available?)

(deftest generated-uneven-map-offset-replay-and-profiled-device-completion
  (if (device-present?)
    (do
      (runtime/initialize! {:include-paths [(str (System/getenv "CUDA_HOME") "/include")]})
      (let [info (runtime/execution-device-info) n 259 target :cuda:runtime-device-test
            _ (hardware/register-target-device!
               target {:type :cuda :name (:name info)
                       :capabilities {:subgroup-sizes [(:warp-size info)] :warp-size (:warp-size info)
                                      :max-workgroup-size (:max-workgroup-size info)
                                      :shared-local-memory (:shared-local-memory info)
                                      :compute-capability (:compute-capability info)}})
            compilation (equation/compile #'fixtures/public-c-family-map {:target target :dtype :float})
            kernel (first (:kernels compilation))
            input (runtime/make-buffer (+ n 9) :float)
            output (runtime/make-buffer (+ n 11) :float)
            in-view (runtime/slice-buffer input 12 (* n 4) :float)
            out-view (runtime/slice-buffer output 20 (* n 4) :float)
            arena (runtime/make-kernel-arena!)]
        (try
          (is (= :cuda-c (:target kernel)))
          (is (= :portable-segmap (get-in kernel [:attributes :kernel-body :attributes :kind])))
          (is (= :none (get-in compilation [:stats :fallback])))
          (is (some? (equation/lower compilation [(float-array n) n])) "shared LinkPlan lowering")
          (runtime/register-kernel! (:kernel-name kernel) kernel arena)
          (let [args (mapv (fn [slot]
                             (case (:kind slot)
                               :input in-view :output out-view
                               :scalar {:type (:kernel-dtype slot) :value n})) (:abi kernel))
                prepared (runtime/bind-kernel-call (call/make kernel args))
                graph (runtime/record-graph! [prepared] {:profile? true})]
            (try
              (doseq [pass [0 1]]
                (let [host (float-array (map #(+ (* pass 7.0) (/ % 4.0)) (range n)))
                      fixture-result (fixtures/public-c-family-map host n)
                      ;; Independent direct JVM arithmetic, separate from Raster's dispatch/compiler.
                      oracle (mapv #(float (* 2.0 (double %))) host)]
                  (runtime/array->buffer! input (float-array (repeat (+ n 9) -999)))
                  (runtime/array->buffer! output (float-array (repeat (+ n 11) -777)))
                  (runtime/array->buffer! in-view host)
                  (let [token (runtime/submit-graph! graph)]
                    (runtime/await-event! token)
                    (is (runtime/event-complete? token))
                    (runtime/release-event! token))
                  (is (= oracle (vec fixture-result)) "existing fixture's JVM parity")
                  (is (= oracle (vec (runtime/buffer->array out-view))) "independent direct JVM arithmetic")
                  (let [all (vec (runtime/buffer->array output))
                        timestamps (runtime/read-graph-timestamps! graph)]
                    (is (= (repeat 5 (float -777)) (take 5 all)))
                    (is (= (repeat 6 (float -777)) (drop (+ n 5) all)))
                    (is (<= 0.0 (:wall-ms timestamps)))
                    (is (= 1 (count (:kernels timestamps))))
                    (println "[CUDA DEVICE]" {:pass pass :device info :nvrtc-version (nvrtc/version)
                                              :target (:compilation-identity prepared)
                                              :elements n :input-byte-offset 12 :output-byte-offset 20
                                              :timing timestamps}))))
              ;; Host fault tests established these guards first. On the real device this
              ;; deliberately exercises only rejected cleanup, never an unsafe native free.
              (let [token (runtime/submit-graph! graph)]
                (doseq [[resource reason] [[prepared :cuda-prepared-in-use] [input :cuda-buffer-in-use]]]
                  (let [error (try (cleanup/release! (::cleanup/owner resource)) nil
                                   (catch clojure.lang.ExceptionInfo e (ex-data e)))]
                    (is (= reason (:reason error)))
                    (is (true? (:cleanup-retry-safe? error)))))
                (is (.isAlive (.scope (:parameters prepared))))
                (runtime/await-event! token)
                (runtime/release-event! token)
                (is (= (mapv #(float (* 2.0 (+ 7.0 (/ % 4.0)))) (range n))
                       (vec (runtime/buffer->array out-view)))))
              (finally
                (runtime/destroy-graph! graph)
                (runtime/destroy-prepared! prepared)
                (is (empty? (cleanup/pending (::cleanup/owner graph))))
                (is (empty? (cleanup/pending (::cleanup/owner prepared)))))))
          (finally
            (runtime/free-buffer! input) (runtime/free-buffer! output)
            (runtime/close-kernel-arena! arena)
            (is (empty? (cleanup/pending (::cleanup/owner input))))
            (is (empty? (cleanup/pending (::cleanup/owner output))))))))
    (device-probe/cuda-skip! "No CUDA device for optional runtime proof")))
