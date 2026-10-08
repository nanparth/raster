(ns raster.gpu.cuda-transfer-device-test
  "Required-device pinned async transfer proof through public session contracts."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-driver-probe :as probe]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.runtime-root :as root]))

(defn- completed? [session event]
  (loop [remaining 10000]
    (cond (gpu/event-complete? session event) true
          (zero? remaining) false
          :else (do (Thread/sleep 1) (recur (dec remaining))))))

(defn- input-array [dt n pass]
  (if (= :float dt)
    (float-array (map #(float (+ pass (/ % 4.0))) (range n)))
    ;; Raw IEEE binary16 storage includes signed zero, subnormal, normal and infinity.
    (short-array (take n (cycle (if (zero? pass)
                                [0 -32768 1 15360 -17408 31744]
                                [1024 16384 -16384 2 0 -32768]))))))

(defn- sentinel-array [dt n]
  (if (= :float dt) (float-array (repeat n -777)) (short-array (repeat n 7))))

(defn- mutate-first! [dt array]
  (if (= :float dt) (aset array 0 (float 99)) (aset array 0 (short 99))))

(defn- prove-transfers! [dt]
  (let [n 257 width (if (= :float dt) 4 2) session (gpu/make-session :cuda:0)]
    (try
      (gpu/alloc! session {:x [dt (+ n 9) nil]})
      (let [view (gpu/buffer-view session :x {:byte-offset (* 3 width) :shape [n]})
            input (input-array dt n 0) expected (vec input) output (sentinel-array dt n)
            sentinel (vec output)
            upload (gpu/submit-upload-ranges! session [[view input {:elements n}]])
            download (gpu/submit-download-ranges! session [[view output {:elements n}]])]
        (mutate-first! dt input)
        (is (completed? session download) "native event polling reaches completion")
        (is (= sentinel (vec output)) "completed polling does not publish staged downloads")
        (gpu/await-event! session upload)
        (is (= [output] (gpu/await-event! session download)) "public await returns its host destinations")
        (let [measurement (gpu/event-measurement session download)]
          (is (= :device-event (:timing-source measurement)))
          (is (= (* n width) (:bytes measurement)))
          (is (true? (:asynchronous? measurement))))
        (is (= expected (vec output)) "upload source was snapshotted before return")
        (gpu/release-event! session upload)
        (gpu/release-event! session download)
        (let [changed (input-array dt n 1) changed-oracle (vec changed)
              replay (sentinel-array dt n)
              up (gpu/submit-upload-ranges! session [[view changed {:elements n}]])
              down (gpu/submit-download-ranges! session [[view replay {:elements n}]])]
          (mutate-first! dt changed)
          (gpu/release-event! session down)
          (gpu/release-event! session up)
          (is (= changed-oracle (vec replay)) "release without await publishes the changed snapshot")
          (let [event-count (count (:events @session))
                invalid-error (try
                                (gpu/submit-upload-ranges! session
                                                          [[view input {:elements n}]
                                                           [view input {:elements (inc n)}]])
                                nil (catch clojure.lang.ExceptionInfo error error))]
            (is (some? invalid-error))
            (is (= event-count (count (:events @session))))
          (let [unchanged (sentinel-array dt n)
                check (gpu/submit-download-ranges! session [[view unchanged {:elements n}]])]
            (gpu/release-event! session check)
            (is (= changed-oracle (vec unchanged)) "invalid final entry prevents the entire upload batch")))
        (let [zero (gpu/submit-download-ranges! session [[view output {:elements 0}]])]
          (is (gpu/event-complete? session zero))
          (gpu/await-event! session zero)
          (is (= 0 (:bytes (gpu/event-measurement session zero))))
          (gpu/release-event! session zero))
        (let [last-input (input-array dt n 0) last-oracle (vec last-input)
              closing-output (sentinel-array dt n)]
          (gpu/submit-upload-ranges! session [[view last-input {:elements n}]])
          (gpu/submit-download-ranges! session [[view closing-output {:elements n}]])
          (gpu/close-session! session)
          (is (= last-oracle (vec closing-output)) "session close drains and publishes outstanding transfers")
          (is (= :closed (:lifecycle @session)))
          (is (= 0 (root/lease-count runtime/state))))
        (println "[CUDA ASYNC TRANSFER]" {:dtype dt :elements n :byte-offset (* 3 width)
                                             :device (runtime/execution-device-info)
                                             :physical-transfer-queue :existing-stream})))
      (finally (gpu/close-session! session)))))

(deftest pinned-async-uneven-float-and-encoded-half-public-contracts
  (if (= :ready (:status (probe/probe)))
    (doseq [dt [:float :half]] (prove-transfers! dt))
    (if (= "1" (System/getenv "RASTER_REQUIRE_CUDA_DEVICE"))
      (is false "CUDA device is required; async transfer proof cannot skip")
      (println "[CUDA SKIP] No device for optional pinned async transfer proof"))))
