(ns raster.gpu.cuda-copy-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.cuda-lifecycle-test :as fixture]))

(deftest public-resident-copy-uses-view-offsets-and-returns-identical-destination
  (#'fixture/with-driver
   (fn [ops _]
     (let [session (gpu/make-session :cuda:0)]
       (try
         (gpu/alloc! session {:source [:float 20 nil] :destination [:float 20 nil]})
         (let [source (gpu/buffer-view session :source {:byte-offset 12 :shape [5]})
               destination (gpu/buffer-view session :destination {:byte-offset 28 :shape [5]})
               before (count @ops)
               result (try (gpu/copy-range! session source destination {:elements 5})
                           (catch clojure.lang.ExceptionInfo _ nil))]
           (is (identical? destination result) "resident D2D copy must return the original destination view")
           (when result
             (is (= [:copy :stream-sync] (mapv first (drop before @ops))))
             (let [[_ [dst src bytes]] (nth @ops before)]
               (is (= (+ 28 (:device-pointer (gpu/buffer session :destination))) dst))
               (is (= (+ 12 (:device-pointer (gpu/buffer session :source))) src))
               (is (= 20 bytes)))))
         (finally (gpu/close-session! session)))))))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest copy-validation-zero-and-overlap-reject-before-driver-contact
  (#'fixture/with-driver
   (fn [ops _]
     (let [session (gpu/make-session :cuda:0) other (gpu/make-session :cuda:0)]
       (try
         (gpu/alloc! session {:x [:float 20 nil] :half [:half 20 nil]})
         (gpu/alloc! other {:x [:float 20 nil]})
         (let [left (gpu/buffer-view session :x {:byte-offset 4 :shape [5]})
               overlap (gpu/buffer-view session :x {:byte-offset 12 :shape [5]})
               disjoint (gpu/buffer-view session :x {:byte-offset 40 :shape [5]})
               foreign (gpu/buffer-view other :x) before (count @ops)]
           (doseq [[src dst spec] [[left overlap {:elements 5}] [left left {:elements 5}]
                                  [left :half {:elements 1}] [left foreign {:elements 1}]
                                  [left disjoint {:elements 6}] [left disjoint {:elements -1}]]]
             (is (some? (error-of #(gpu/copy-range! session src dst spec)))))
           (is (= before (count @ops)))
           (is (identical? left (gpu/copy-range! session left left {:elements 0})))
           (is (= before (count @ops)))
           (is (identical? disjoint (gpu/copy-range! session left disjoint {:elements 5})))
           (doseq [range [[-1 0 1] [0 -1 1] [0 0 -1] [20 0 1] [0 20 1]]]
             (let [before (count @ops) b (gpu/buffer session :x)]
               (is (= :cuda-copy-range (:reason (ex-data (error-of #(apply runtime/copy-buffer-range! b b range))))))
               (is (= before (count @ops))))))
         (finally (gpu/close-session! session) (gpu/close-session! other)))))))

(deftest copy-unknown-completion-keeps-both-buffers-and-root-with-one-native-wait
  (#'fixture/with-driver
   (fn [ops fault]
     (let [src (runtime/make-buffer 4 :float) dst (runtime/make-buffer 4 :float)
           failure (ex-info "copy completion unknown" {})]
       (reset! fault {:operation :stream-sync :error failure})
       (is (identical? failure (error-of #(runtime/copy-buffer-range! src dst 0 0 4))))
       (is (= 1 (count (filter #(= :stream-sync (first %)) @ops))))
       (is (= 3 (root/lease-count runtime/state)))
       (is (= 1 (count @(:debts (:prepared-execution @runtime/state)))))
       (doseq [b [src dst]]
         (is (= :cuda-buffer-in-use (:reason (ex-data (error-of #(cleanup/release! (::cleanup/owner b))))))))
       (reset! fault nil)
       (let [before (count @ops)]
         (is (identical? failure (error-of runtime/synchronize-async!)))
         (is (= before (count @ops))))))))

(deftest copy-explicitly-retry-safe-completion-can-retire-retained-debt
  (#'fixture/with-driver
   (fn [_ fault]
     (let [src (runtime/make-buffer 4 :half) dst (runtime/make-buffer 4 :half)
           failure (ex-info "copy drain retry allowed" {:cleanup-retry-safe? true})]
       (reset! fault {:operation :stream-sync :error failure})
       (is (identical? failure (error-of #(runtime/copy-buffer-range! src dst 0 0 4))))
       (is (= 3 (root/lease-count runtime/state)))
       (reset! fault nil)
       (runtime/synchronize-async!)
       (is (empty? @(:debts (:prepared-execution @runtime/state))))
       (is (= 2 (root/lease-count runtime/state)))
       (runtime/free-buffer! src) (runtime/free-buffer! dst)))))
