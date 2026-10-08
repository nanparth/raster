(ns raster.gpu.cuda-unsupported-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-runtime :as runtime]))

(deftest unimplemented-routes-identify-backend-and-operation-without-driver-contact
  (binding [driver/*invoke* (fn [& _] (throw (ex-info "unexpected native contact" {})))]
    (doseq [operation '[invoke-registered-kernel invoke-registered-map-void-kernel
                       bind-registered-map-void-kernel register-kernel-dispatch!
                       kernel-dispatch-registry-entry copy-buffer-range!]]
      (let [function (ns-resolve 'raster.gpu.cuda-runtime operation)]
        (is (some? function) (str operation " must report an explicit unsupported route"))
        (when function
          (let [data (try (function) nil (catch clojure.lang.ExceptionInfo e (ex-data e)))]
            (is (= :cuda-unsupported-operation (:reason data)))
            (is (= :cuda (:backend data)))
            (is (= (keyword (name operation)) (:operation data)))))))))

(deftest transfer-capabilities-describe-proven-async-work-on-the-existing-stream
  (let [capabilities (ns-resolve 'raster.gpu.cuda-runtime 'transfer-capabilities)]
    (is (some? capabilities))
    (when capabilities
      (let [facts (capabilities)]
        (is (true? (:async-h2d? facts)))
        (is (true? (:async-d2h? facts)))
        (is (= :device-event (:submission facts)))
        (is (= :runtime-owned-native (:host-staging facts)))
        (is (= :pinned (:staging-memory facts)))
        (is (true? (:host-lease-until-await? facts)))
        (is (true? (:physically-serialized? facts)))
        (is (false? (:independent-physical-queue? facts)))))))
