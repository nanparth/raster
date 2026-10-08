(ns raster.gpu.cuda-execution
  "Conservative CUDA logical graph replay and native completion ownership.
   Prepared calls retain ABI/storage facts; this leaf only orders launches/events."
  (:require [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]))

(defn- owner! [value]
  (or (::cleanup/owner value)
      (throw (ex-info "CUDA execution value has no cleanup owner" {:reason :missing-cleanup-owner}))))

(defn- live! [graph]
  (cleanup/assert-live! (owner! graph))
  (doseq [prepared (:prepareds graph)] (cleanup/assert-live! (owner! prepared))) graph)

(defn- retire-submission! [graph]
  (when-let [submission @(:submission-state graph)]
    (cleanup/release! (owner! submission))
    (when (empty? (cleanup/pending (owner! submission)))
      (compare-and-set! (:submission-state graph) submission nil))))

(defn record-graph! [state prepareds {:keys [profile? adopt-cleanup!] :or {profile? false}}]
  (when-not (vector? prepareds)
    (throw (ex-info "CUDA recorded calls require an ordered vector" {:reason :cuda-graph-order})))
  (doseq [p prepareds] (cleanup/assert-live! (owner! p)))
  (let [submission-state (atom nil) pins (atom []) token (random-uuid)
        graph {:prepareds prepareds :submission-state submission-state :profile? profile?}]
    (root/construct-child!
     state [{:id :submission :release #(retire-submission! graph)}
            {:id :prepared-bindings :after #{:submission}
             :release #(doseq [borrow @pins] (swap! borrow disj token))}]
     (fn [_ _]
       (doseq [p prepareds]
         (swap! pins conj (:graph-borrows p))
         (swap! (:graph-borrows p) conj token))
       graph)
     adopt-cleanup!)))

(defn submit-graph! [state graph native!]
  (live! graph)
  (if (empty? (:prepareds graph))
    {:complete? true}
  (let [prepareds (:prepareds graph) started (atom false) completed (atom false)
        stream (:stream @state)
        event-count (if (:profile? graph) (* 2 (count prepareds)) 1)
        slots (mapv (fn [_] (cleanup/acquisition-slot)) (range event-count))
        owner (cleanup/owner
               (into [{:id :completion
                       :release #(when (and @started (not @completed))
                                   (native! :stream-sync [stream]) (reset! completed true))}]
                     (map-indexed
                      (fn [i slot]
                        {:id [:event i] :after #{:completion}
                         :release #(cleanup/release-native! slot (fn [e] (native! :event-destroy [e])))}) slots)))
        submission {::cleanup/owner owner :completed? completed :started? started
                    :stream stream :graph graph :profile? (:profile? graph)}
        submission-state (:submission-state graph)]
    (when-not (compare-and-set! submission-state nil submission)
      (throw (ex-info "CUDA graph still owns its previous submission" {:reason :submission-still-owned})))
    (try
      (cleanup/build!
       owner
       (fn []
         (let [events (mapv #(cleanup/acquire-native! % (fn [] (native! :event-create []))) slots)]
           (doseq [[index prepared] (map-indexed vector prepareds)]
             (when (:profile? graph) (native! :event-record [(nth events (* index 2)) stream]))
             (reset! started true)
             (native! :launch [(:function prepared) (:geometry prepared) (:parameters prepared) stream])
             (when (:profile? graph) (native! :event-record [(nth events (inc (* index 2))) stream])))
           (when-not (:profile? graph)
             (reset! started true)
             (native! :event-record [(peek events) stream]))
           (let [result (assoc submission :events events :event (peek events))]
             (reset! submission-state result) result)))
       ;; Graph publication already retains this canonical owner if rollback fails.
       (fn [_] nil))
      (catch Throwable primary
        (when (empty? (cleanup/pending owner)) (reset! submission-state nil))
        (throw primary))))))

(defn event-complete? [token native!]
  (if (:complete? token)
    true
    (do
  (cleanup/assert-live! (owner! token))
  (let [complete (boolean (native! :event-query [(:event token)]))]
    (when complete (reset! (:completed? token) true)) complete))))

(defn await-event! [token native!]
  (when-not (:complete? token)
  (cleanup/assert-live! (owner! token))
  (when-not @(:completed? token)
    (native! :event-sync [(:event token)])
    (reset! (:completed? token) true)))
  nil)

(defn release-event! [token]
  (when-not (:complete? token)
  (owner! token)
  (when-not (:profile? token)
    (cleanup/release! (owner! token))
    (let [slot (:submission-state (:graph token))]
      (when (and slot (identical? (owner! token) (::cleanup/owner @slot)))
        (reset! slot nil))))) nil)

(defn destroy-graph! [graph] (cleanup/release! (owner! graph)))
(defn reset-graph-events! [graph] (live! graph) (retire-submission! graph) nil)

(defn read-graph-timestamps! [graph native!]
  (live! graph)
  (when-not (:profile? graph)
    (throw (ex-info "CUDA graph was not recorded with profiling" {:reason :cuda-graph-not-profiled})))
  (let [submission @(:submission-state graph)]
    (when-not submission
      (throw (ex-info "CUDA graph has no profiled submission" {:reason :cuda-graph-no-submission})))
    (await-event! submission native!)
    (let [events (:events submission)
          rows (mapv (fn [i p]
                       (let [ns (native! :event-elapsed [(nth events (* i 2)) (nth events (inc (* i 2)))])]
                         {:kernel-name (:kernel-name p) :phase (:phase p)
                          :ms (/ (double ns) 1000000.0) :context-ms (/ (double ns) 1000000.0)}))
                     (range (count (:prepareds graph))) (:prepareds graph))
          wall (if (seq rows) (/ (double (native! :event-elapsed [(first events) (peek events)])) 1000000.0) 0.0)]
      (retire-submission! graph)
      {:kernels rows :wall-ms wall :ns-per-tick 1.0})))
