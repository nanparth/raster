(ns raster.gpu.cuda-prepared-async-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-lifecycle-test :as fixture]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn- with-session [f]
  (#'fixture/with-driver
   (fn [ops fault]
     (let [session (gpu/make-session :cuda:0)
           artifact (artifact/make
                     (assoc (#'fixture/kernel)
                            :source "extern \"C\" __global__ void lifetime_contract(float s, float* x, long n) {}"
                            :abi [(abi/slot 's :scalar :float) (abi/slot 'x :inout :float)
                                  (abi/slot 'n :scalar :long :role :bound)]
                            :arguments '[s x n]))]
       (gpu/alloc! session {:x [:float 4 nil]})
       (runtime/register-kernel! (:kernel-name artifact) artifact (:arena-id @session))
       ;; Transport fixture installs a shared KernelArtifact, never a compiler convention.
       (swap! session assoc :kernels {:phase [artifact]})
       (f session ops fault)))))

(deftest public-async-preparation-submits-without-wait-and-sync-establishes-completion
  (with-session
    (fn [session ops _]
      (try
        (is (some? (gpu/prepare! session :synchronous {"x" :x} [(float 2)] 4
                                {:kernel-phase :phase})))
        (gpu/release-prepared! session :synchronous)
        (let [prepared (try (gpu/prepare! session :phase {"x" :x} [(float 2)] 4 {:async? true})
                            (catch clojure.lang.ExceptionInfo _ nil))]
          (is (some? prepared) "working ordered binding must support asynchronous preparation")
          (when prepared
            (let [before (count @ops)]
              (gpu/invoke-bound! session :phase)
              (gpu/invoke-bound! session :phase)
              (is (= [:launch :launch] (mapv first (drop before @ops))))
              (gpu/sync! session)
              (is (= :stream-sync (ffirst (reverse @ops)))))))
        (finally (gpu/close-session! session))))))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))
(defn- prepare! [session key async?]
  (gpu/prepare! session key {"x" :x} [(float 2)] 4 {:async? async? :kernel-phase :phase}))

(deftest synchronous-launch-waits-and-one-sync-settles-multiple-async-bindings
  (with-session
    (fn [session ops _]
      (try
        (prepare! session :sync false)
        (let [before (count @ops)]
          (gpu/invoke-bound! session :sync)
          (is (= [:launch :stream-sync] (mapv first (drop before @ops)))))
        (prepare! session :left true) (prepare! session :right true)
        (gpu/invoke-bound! session :left) (gpu/invoke-bound! session :right)
        (gpu/invoke-bound! session :left)
        (let [before (count @ops)]
          (gpu/sync! session)
          (is (= [:stream-sync] (mapv first (drop before @ops))))
          (gpu/release-prepared! session :left) (gpu/release-prepared! session :right)
          (is (= 1 (count (filter #(= :stream-sync (first %)) (drop before @ops))))))
        (finally (gpu/close-session! session))))))

(deftest canonical-prepared-retirement-drains-before-parameters-and-module
  (with-session
    (fn [session ops _]
      (try
        (let [prepared (prepare! session :phase true)]
          (gpu/invoke-bound! session :phase)
          (is (.isAlive (.scope (:parameters prepared))))
          (let [before (count @ops)]
            (cleanup/release! (::cleanup/owner prepared))
            (is (= [:stream-sync :module-unload] (mapv first (drop before @ops))))
            (is (not (.isAlive (.scope (:parameters prepared)))))))
        (finally (gpu/close-session! session))))))

(deftest public-reprepare-drains-old-pending-binding-before-publishing-new
  (with-session
    (fn [session ops _]
      (try
        (let [old (prepare! session :phase true)]
          (gpu/invoke-bound! session :phase)
          (let [before (count @ops) replacement (prepare! session :phase true)
                operations (mapv first (drop before @ops))]
            (is (not (identical? old replacement)))
            (is (not (.isAlive (.scope (:parameters old)))))
            (is (< (.indexOf operations :stream-sync) (.indexOf operations :module-unload)))
            (is (.isAlive (.scope (:parameters replacement))))))
        (finally (gpu/close-session! session))))))

(deftest unknown-public-sync-is-retained-across-canonical-close-and-replacement
  (with-session
    (fn [session ops fault]
      (let [old (prepare! session :phase true) failure (ex-info "completion unknown" {})]
        (gpu/invoke-bound! session :phase)
        (reset! fault {:operation :stream-sync :error failure})
        (is (identical? failure (error-of #(gpu/sync! session))))
        (reset! fault nil)
        (let [before (count @ops)]
          (is (identical? failure (error-of #(gpu/release-prepared! session :phase))))
          (is (= before (count @ops))))
        (is (.isAlive (.scope (:parameters old))))
        (is (seq (cleanup/pending (::cleanup/owner old))))
        (let [before (count @ops)]
          (is (identical? failure (error-of #(prepare! session :phase true))))
          (is (= before (count @ops)) "poison rejects replacement before candidate acquisition")
          (is (not-any? #(= :stream-sync (first %)) (drop before @ops))))
        (let [before (count @ops)]
          (is (some? (error-of #(gpu/invoke-bound! session :phase))))
          (is (= before (count @ops))))
        (is (.isAlive (.scope (:parameters old))))))))

(deftest explicitly-retry-safe-completion-can-drain-and-retire-closing-preparation
  (with-session
    (fn [session ops fault]
      (let [prepared (prepare! session :phase true)
            failure (ex-info "wait safe to retry" {:cleanup-retry-safe? true})]
        (gpu/invoke-bound! session :phase)
        (reset! fault {:operation :stream-sync :error failure})
        (is (identical? failure (error-of #(cleanup/release! (::cleanup/owner prepared)))))
        (let [before (count @ops)]
          (is (some? (error-of #(gpu/invoke-bound! session :phase))))
          (is (= before (count @ops))))
        (is (.isAlive (.scope (:parameters prepared))))
        (reset! fault nil)
        (gpu/release-prepared! session :phase)
        (is (not (.isAlive (.scope (:parameters prepared)))))
        (gpu/close-session! session)))))

(deftest unknown-launch-outcome-retains-prepared-use-until-safe-drain
  (with-session
    (fn [session ops fault]
      (let [prepared (prepare! session :phase true) failure (ex-info "launch outcome unknown" {})]
        (reset! fault {:operation :launch :error failure})
        (is (identical? failure (error-of #(gpu/invoke-bound! session :phase))))
        (is (.isAlive (.scope (:parameters prepared))))
        (reset! fault nil)
        (let [before (count @ops)]
          (is (identical? failure (error-of #(gpu/invoke-bound! session :phase))))
          (is (= before (count @ops))))
        (gpu/release-prepared! session :phase)
        (is (not (.isAlive (.scope (:parameters prepared)))))
        (gpu/close-session! session)))))

(deftest unknown-prepared-completion-blocks-whole-and-ranged-host-transfers
  (doseq [operation [(fn [s h] (gpu/upload! s :x h))
                     (fn [s _] (gpu/download s :x))
                     (fn [s h] (gpu/upload-range! s :x h {:elements 4}))
                     (fn [s h] (gpu/download-range! s :x h {:elements 4}))]]
    (with-session
      (fn [session ops fault]
        (prepare! session :phase true)
        (gpu/invoke-bound! session :phase)
        (let [failure (ex-info "completion outcome unknown" {}) host (float-array [-7 -7 -7 -7])]
          (reset! fault {:operation :stream-sync :error failure})
          (is (identical? failure (error-of #(gpu/sync! session))))
          (reset! fault nil)
          (let [before (count @ops)]
            (is (identical? failure (error-of #(operation session host))))
            (is (= before (count @ops))))
          (is (= [-7.0 -7.0 -7.0 -7.0] (vec host))))))))

(deftest unknown-prepared-completion-blocks-graph-and-async-range-admission
  (with-session
    (fn [session ops fault]
      (let [prepared (prepare! session :phase true) graph (runtime/record-graph! [prepared])
            failure (ex-info "completion outcome unknown" {}) host (float-array 4)]
        (gpu/invoke-bound! session :phase)
        (reset! fault {:operation :stream-sync :error failure})
        (is (identical? failure (error-of #(gpu/sync! session))))
        (reset! fault nil)
        (let [before (count @ops)]
          (is (identical? failure (error-of #(runtime/submit-graph! graph))))
          (is (identical? failure (error-of #(gpu/submit-upload-ranges! session [[:x host {:elements 4}]]))))
          (is (identical? failure (error-of #(gpu/submit-download-ranges! session [[:x host {:elements 4}]]))))
          (is (= before (count @ops))))
        (runtime/destroy-graph! graph)))))

(deftest explicitly-retry-safe-prepared-sync-recovers-through-synchronous-transfer
  (with-session
    (fn [session ops fault]
      (try
        (prepare! session :phase true)
        (gpu/invoke-bound! session :phase)
        (let [failure (ex-info "completion retry permitted" {:cleanup-retry-safe? true}) host (float-array 4)]
          (reset! fault {:operation :stream-sync :error failure})
          (is (identical? failure (error-of #(gpu/sync! session))))
          (reset! fault nil)
          (let [before (count @ops)]
            (is (identical? host (gpu/download-range! session :x host {:elements 4})))
            (is (= [:stream-sync :download] (mapv first (drop before @ops))))))
        (finally (gpu/close-session! session))))))

(deftest indeterminate-completion-rejects-allocation-but-permits-proven-retirement
  (with-session
    (fn [session ops fault]
      (let [completed (prepare! session :completed false) pending (prepare! session :pending true)
            unrelated (runtime/make-buffer 4 :float) failure (ex-info "completion outcome unknown" {})]
        (gpu/invoke-bound! session :completed)
        (gpu/invoke-bound! session :pending)
        (reset! fault {:operation :stream-sync :error failure})
        (is (identical? failure (error-of #(gpu/sync! session))))
        (reset! fault nil)
        (let [before (count @ops)]
          (is (identical? failure (error-of #(runtime/make-buffer 4 :float))))
          (is (= before (count @ops)))
          (cleanup/release! (::cleanup/owner completed))
          (runtime/free-buffer! unrelated)
          (is (= [:module-unload :free] (mapv first (drop before @ops)))))
        (is (not (.isAlive (.scope (:parameters completed)))))
        (is (.isAlive (.scope (:parameters pending))))))))

(deftest concurrent-transfer-admission-cannot-slip-past-failing-completion
  (with-session
    (fn [session ops _]
      (prepare! session :phase true)
      (gpu/invoke-bound! session :phase)
      (let [native driver/*invoke* entered (promise) resume (promise) admission-started (promise)
            failure (ex-info "completion outcome unknown" {}) before (count @ops)]
        (binding [driver/*invoke* (fn [operation arguments]
                                   (if (= :stream-sync operation)
                                     (do (swap! ops conj [operation arguments])
                                         (deliver entered true)
                                         (when (= ::timeout (deref resume 5000 ::timeout))
                                           (throw (ex-info "test completion barrier timed out" {})))
                                         (throw failure))
                                     (native operation arguments)))]
          (let [completion (future (error-of #(gpu/sync! session)))]
            (try
              (is (= true (deref entered 5000 ::timeout)))
              (let [admission (future (deliver admission-started true)
                                      (error-of #(gpu/upload-range! session :x (float-array 4) {:elements 4})))]
                (is (= true (deref admission-started 5000 ::timeout)))
                (is (= ::waiting (deref admission 50 ::waiting)))
                (deliver resume true)
                (is (identical? failure (deref completion 5000 ::timeout)))
                (is (identical? failure (deref admission 5000 ::timeout)))
                (is (= [:stream-sync] (mapv first (drop before @ops)))))
              (finally (deliver resume true)))))))))

(deftest nonpoisoned-replacement-destruction-failure-retires-candidate-and-retains-old-debt
  (with-session
    (fn [session ops _]
      (let [old (prepare! session :phase true) native driver/*invoke*
            first-unload? (atom true) failure (ex-info "old module destruction unknown" {})]
        (gpu/invoke-bound! session :phase)
        (binding [driver/*invoke* (fn [operation arguments]
                                   (if (and (= :module-unload operation)
                                            (compare-and-set! first-unload? true false))
                                     (do (swap! ops conj [operation arguments]) (throw failure))
                                     (native operation arguments)))]
          (let [before (count @ops)]
            (is (identical? failure (error-of #(prepare! session :phase true))))
            (is (= 2 (count (filter #(= :module-unload (first %)) (drop before @ops))))
                "failed old destruction and successful candidate rollback are separate")
            (is (not (.isAlive (.scope (:parameters old)))))
            (is (seq (cleanup/pending (::cleanup/owner old))))
            (is (= 1 (count (:prepared @session))))
            (is (nil? @(:completion-error (:prepared-execution @runtime/state))))
            (let [unrelated (runtime/make-buffer 4 :float)] (runtime/free-buffer! unrelated))
            (let [before (count @ops)]
              (is (identical? failure (error-of #(gpu/close-session! session))))
              (is (= before (count @ops)) "indeterminate old destruction is never repeated"))))))))
