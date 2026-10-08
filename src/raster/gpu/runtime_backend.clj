(ns raster.gpu.runtime-backend
  "The supported resident GPU runtime backends. Compiler targets are broader:
   HIP can emit code without being registered here as an executable session."
  (:require [clojure.string :as str]))

(def ^:private resident-backends
  {:ze {:namespace 'raster.gpu.ze-runtime
        :kernel-body-c-dialect :opencl-portable
        :memory-space :shared
        :coherence :host-coherent
        :owned-slice? false}
   :ocl {:namespace 'raster.gpu.ocl-runtime
         :kernel-body-c-dialect :opencl-portable
         :memory-space :device
         :coherence :explicit-transfer
         :owned-slice? true}
   :cuda {:namespace 'raster.gpu.cuda-runtime
          :kernel-body-c-dialect :cuda
          :memory-space :device
          :coherence :explicit-transfer
          :owned-slice? false}})

(defn backend-type
  "Return the resident backend of a device ID, rejecting compile-only targets."
  [device-id]
  (let [s (when (keyword? device-id) (name device-id))
        backend (cond
                  (and s (str/starts-with? s "ze")) :ze
                  (and s (str/starts-with? s "ocl")) :ocl
                  (and s (str/starts-with? s "cuda")) :cuda)]
    ;; Session construction passes no ordinal to the singleton runtime. Only the
    ;; default device is selectable until the session contract carries that choice.
    (when (and (= :cuda backend) (not= :cuda:0 device-id))
      (throw (ex-info "CUDA resident sessions currently support only :cuda:0"
                      {:reason :cuda-runtime-device-unsupported
                       :device-id device-id :supported-devices [:cuda:0]})))
    (if (and backend (contains? resident-backends backend))
      backend
      (throw (ex-info "No resident GPU runtime for device; use :ze:N, :ocl:N or :cuda:0"
                      {:device-id device-id
                       :supported-backends (vec (sort (keys resident-backends)))
                       :compile-only? (boolean (and s (re-find #"^hip:" s)))})))))

(defn descriptor
  "Return runtime facts used by the session and value layers."
  [device-id]
  (get resident-backends (backend-type device-id)))

(defn runtime-namespace
  [device-id]
  (:namespace (descriptor device-id)))
