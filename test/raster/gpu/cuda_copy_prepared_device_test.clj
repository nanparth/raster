(ns raster.gpu.cuda-copy-prepared-device-test
  "Public copy/prepared runtime proof; artifact installation is an explicit test fixture."
  (:require [raster.gpu.device-probe :as device-probe]
            [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-compile-fixtures :as fixtures]
            [raster.compiler.equation-first :as equation]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]))

(defn- with-native-spy [f]
  (let [operations (atom [])]
    (binding [driver/*invoke* (fn [operation args]
                              (swap! operations conj operation)
                              (binding [driver/*invoke* nil] (driver/invoke! operation args)))]
      (f operations))))

(defn- error-of [f] (try (f) nil (catch clojure.lang.ExceptionInfo error error)))
(defn- values [dt n]
  (if (= :float dt) (float-array (map #(float (/ % 4.0)) (range n)))
      (short-array (take n (cycle [0 -32768 1 15360 -17408 31744])))))
(defn- filled [dt n sentinel]
  (if (= :float dt) (float-array (repeat n sentinel)) (short-array (repeat n sentinel))))

(defn- copy-proof! [dt]
  (with-native-spy
    (fn [ops]
      (let [n 257 width (if (= :float dt) 4 2) total (+ n 13)
            session (gpu/make-session :cuda:0) foreign (gpu/make-session :cuda:0)
            source-data (values dt total) sentinel (if (= :float dt) (float -777) (short 7))]
        (try
          (gpu/alloc! session {:source [dt total source-data]
                              :destination [dt total (filled dt total sentinel)]
                              :other [(if (= :float dt) :half :float) total nil]})
          (gpu/alloc! foreign {:x [dt total nil]})
          (let [source (gpu/buffer-view session :source {:byte-offset (* 3 width) :shape [n]})
                destination (gpu/buffer-view session :destination {:byte-offset (* 7 width) :shape [n]})
                before (count @ops)]
            (is (identical? destination (gpu/copy-range! session source destination {:elements n})))
            (is (= [:copy :stream-sync] (vec (drop before @ops))) "direct resident D2D, without host staging")
            (is (= (vec (concat (repeat 7 sentinel) (take n (drop 3 source-data)) (repeat 6 sentinel)))
                   (vec (gpu/download session :destination))))
            (let [before (count @ops)]
              (is (identical? destination (gpu/copy-range! session source destination {:elements 0})))
              (doseq [[src dst spec] [[source destination {:elements (inc n)}]
                                     [source :other {:elements 1}]
                                     [source (gpu/buffer-view foreign :x) {:elements 1}]
                                     [source source {:elements 1}]
                                     [source (gpu/buffer-view session :source {:byte-offset (* 4 width) :shape [n]})
                                      {:elements 5}]]]
                (is (some? (error-of #(gpu/copy-range! session src dst spec)))))
              (is (= before (count @ops)) "all unsupported/invalid copies reject before native contact"))
            (let [left (gpu/buffer-view session :source {:shape [4]})
                  right (gpu/buffer-view session :source {:byte-offset (* (+ n 5) width) :shape [4]})]
              (is (identical? right (gpu/copy-range! session left right {:elements 4})))
              (let [host (filled dt 4 sentinel)]
                (gpu/download-range! session right host {:elements 4})
                (is (= (vec (take 4 source-data)) (vec host)))))
            (println "[CUDA RESIDENT COPY]" {:dtype dt :elements n :source-byte-offset (* 3 width)
                                                :destination-byte-offset (* 7 width)
                                                :device (runtime/execution-device-info)}))
          (finally (gpu/close-session! session) (gpu/close-session! foreign)))
        (is (= 0 (root/lease-count runtime/state)))))))

(deftest required-resident-copy-float-and-encoded-half
  (if @device-probe/cuda-available?
    (doseq [dt [:float :half]] (copy-proof! dt))
    (device-probe/cuda-skip! "Resident copy proof")))

(deftest required-generated-map-public-prepared-async-replay-and-retirement
  (if @device-probe/cuda-available?
    (with-native-spy
      (fn [ops]
        (let [n 259 session (gpu/make-session :cuda:0)
              compilation (equation/compile #'fixtures/public-c-family-map {:target :cuda:0 :dtype :float})
              kernel (first (:kernels compilation))]
          (try
            (is (= :none (get-in compilation [:stats :fallback])))
            (is (= :portable-segmap (get-in kernel [:attributes :kernel-body :attributes :kind])))
            (gpu/alloc! session {:input [:float (+ n 9) (float-array (repeat (+ n 9) -999))]
                                :output [:float (+ n 11) (float-array (repeat (+ n 11) -777))]})
            (let [in-view (runtime/slice-buffer (gpu/buffer session :input) 12 (* n 4) :float)
                  out-view (runtime/slice-buffer (gpu/buffer session :output) 20 (* n 4) :float)
                  bindings (into {} (for [slot (:abi kernel) :when (not= :scalar (:kind slot))]
                                      [(name (:name slot)) (case (:kind slot) :input :in-view :output :out-view)]))
                  prepare (fn [phase]
                            (gpu/prepare! session phase bindings [] n {:async? true :kernel-phase :map}))]
              (gpu/register-buffer! session :in-view in-view)
              (gpu/register-buffer! session :out-view out-view)
              ;; This fixture installs an already equation-generated shared artifact. It
              ;; does not claim the historical compile! API or application parity.
              (runtime/register-kernel! (:kernel-name kernel) kernel (:arena-id @session))
              (swap! session assoc :kernels {:map [kernel]})
              (let [prepared (prepare :map) other (prepare :other)]
                (doseq [pass [0 1]]
                  (let [input (float-array (map #(float (+ (* pass 7) (/ % 4.0))) (range n)))
                        oracle (mapv #(float (* 2.0 (double %))) input)]
                    (runtime/array->buffer! in-view input)
                    (let [before (count @ops)]
                      (gpu/invoke-bound! session :map) (gpu/invoke-bound! session :other)
                      (gpu/invoke-bound! session :map)
                      (is (= [:launch :launch :launch] (vec (drop before @ops))))
                      (gpu/sync! session)
                      (is (= [:launch :launch :launch :stream-sync] (vec (drop before @ops)))))
                    (is (= oracle (vec (runtime/buffer->array out-view))))))
                (gpu/invoke-bound! session :map)
                (let [replacement (prepare :map)]
                  (is (not (.isAlive (.scope (:parameters prepared)))))
                  (is (.isAlive (.scope (:parameters replacement))))
                  (gpu/invoke-bound! session :map)
                  (gpu/release-prepared! session :map)
                  (is (not (.isAlive (.scope (:parameters replacement))))))
                (gpu/invoke-bound! session :other)
                (cleanup/release! (::cleanup/owner other))
                (is (not (.isAlive (.scope (:parameters other)))))
                (is (= (mapv #(float (* 2.0 (+ 7.0 (/ % 4.0)))) (range n))
                       (vec (runtime/buffer->array out-view))))
                (let [all (vec (gpu/download session :output))]
                  (is (= (repeat 5 (float -777)) (take 5 all)))
                  (is (= (repeat 6 (float -777)) (drop (+ n 5) all))))
                (let [closing (prepare :closing) before (count @ops)]
                  (gpu/invoke-bound! session :closing)
                  (gpu/close-session! session)
                  (is (not (.isAlive (.scope (:parameters closing)))))
                  (let [operations (vec (drop before @ops))]
                    (is (< (.indexOf operations :stream-sync) (.indexOf operations :free)))))
                (println "[CUDA PREPARED ASYNC]" {:elements n :input-byte-offset 12 :output-byte-offset 20
                                                :device (runtime/execution-device-info)})))
            (finally (gpu/close-session! session)))
          (is (= 0 (root/lease-count runtime/state))))))
    (device-probe/cuda-skip! "Prepared async proof")))
