(ns raster.compiler.backend.gpu.layout-transform
  "Lower target-neutral affine layout-transform schedules to C-family kernel artifacts."
  (:require [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]))

(defn cast-body
  "Build the target-neutral body for one dense representation conversion."
  [{:keys [id kernel-name input output source-dtype destination-dtype vector-width rounding overflow
           extent-dtype]
    :or {vector-width 1 extent-dtype :int}}]
  (schedule/cast-body
   {:id (or id [:layout-transform kernel-name])
    :input input :output output
    :source-dtype source-dtype :destination-dtype destination-dtype
    :vector-width vector-width :extent-dtype extent-dtype
    :rounding rounding :overflow overflow}))

(defn transpose-body
  "Build the target-neutral body for one dense row-major matrix transpose."
  [{:keys [id kernel-name input output element-dtype row-extent-dtype column-extent-dtype]
    :or {row-extent-dtype :int column-extent-dtype :int}}]
  (schedule/transpose-body
   {:id (or id [:layout-transform kernel-name])
    :input input :output output
    :element-dtype element-dtype
    :row-extent-dtype row-extent-dtype :column-extent-dtype column-extent-dtype}))

(defn emit-cast-kernel
  [{:keys [kernel-name input output source-dtype destination-dtype vector-width rounding overflow
           target-dialect extent-dtype]
    :or {vector-width 1 target-dialect :opencl-intel extent-dtype :int}}]
  (let [kernel-body (cast-body
                     {:kernel-name kernel-name :input input :output output
                      :source-dtype source-dtype :destination-dtype destination-dtype
                      :vector-width vector-width :rounding rounding :overflow overflow
                      :extent-dtype extent-dtype})]
    {:source (emitter/emit-scalar-kernel
              kernel-name kernel-body
              {:target-dialect target-dialect
               :parameter-names {input "in" output "out" :layout-elements "n"}})
     :kernel-body kernel-body}))

(defn emit-transpose-kernel
  [{:keys [kernel-name input output element-dtype target-dialect row-extent-dtype column-extent-dtype]
    :or {target-dialect :opencl-intel row-extent-dtype :int column-extent-dtype :int}}]
  (let [kernel-body (transpose-body
                     {:kernel-name kernel-name :input input :output output
                      :element-dtype element-dtype
                      :row-extent-dtype row-extent-dtype :column-extent-dtype column-extent-dtype})]
    (assoc (emitter/emit-scalar-module
            kernel-name kernel-body
            {:target-dialect target-dialect
             :parameter-names {input "in" output "out"
                               :layout-rows "rows" :layout-cols "cols"}})
           :kernel-body kernel-body)))

(defn emit-transpose-artifact
  "Close transpose shape conditions through the checked invocation artifact boundary."
  [{:keys [kernel-name input output target-dialect] :or {target-dialect :opencl-intel} :as options}]
  (let [kernel-body (transpose-body options)
        arguments [input output :layout-rows :layout-cols]
        scheduled (scheduled/make
                   {:source (assoc options :operation :transpose)
                    :body kernel-body :arguments arguments
                    :preconditions (body/required-preconditions kernel-body)
                    :scalar-bindings (scheduled/derive-scalar-bindings kernel-body arguments)
                    :effects {:kind :layout-transform
                              :uses (scheduled/derive-uses kernel-body arguments)}
                    :legality {:kind :dense-affine-transpose}
                    :numerics {:mode :exact :policy :layout-permutation}})]
    (target/emit-artifact kernel-name scheduled target-dialect
                          {:parameter-names {input "input_data" output "output_data"
                                             :layout-rows "rows" :layout-cols "cols"}})))
