(ns raster.compiler.backend.gpu.target
  "Backend-emitter capability predicates.

   Hardware descriptors state what a device can execute. These predicates state which concrete
   source dialect the currently selected backend leaf can emit; schedule-neutral compiler passes
   must not use them to constrain otherwise-portable schedules."
  (:require [clojure.string :as str]
            [raster.compiler.passes.parallel.device :as device]))

(defn kernel-body-c-dialect
  "Select the scalar/control KernelBody source dialect proven by a target descriptor.

   CUDA and HIP backends name their native source dialect directly. OpenCL requires an explicit
   portable subgroup contract unless it is the detected/catalogued Intel path. This predicate is
   about emission legality; it does not claim that a corresponding runtime launcher is installed."
  [desc]
  (let [vendor (some-> (:vendor desc) str str/lower-case)
        matrix-family (get-in desc [:matrix :family])
        backend (:backend desc)
        requested (:subgroup-dialect desc)]
    (cond
      (or (nil? desc) (not= :gpu (:device-type desc))) nil
      (= :cuda backend) :cuda
      (= :hip backend) :hip
      (= :cuda requested) :cuda
      (= :hip requested) :hip
      (= :opencl-portable requested) :opencl-portable
      (and (contains? #{nil :ze :ocl :opencl} backend)
           (or (= :intel-opencl requested)
               (= :dpas matrix-family)
               (and vendor (str/includes? vendor "intel"))))
      :opencl-intel
      :else nil)))

(defn intel-opencl-subgroup-dialect?
  "True when the current OpenCL leaf may emit Intel subgroup attributes and builtins.

   `:subgroup-dialect` is the explicit cross-compile contract. Vendor and DPAS checks retain the
   detected/catalogued Intel path until runtime backend capabilities become a first-class facet."
  [desc]
  (= :opencl-intel (kernel-body-c-dialect desc)))

(defn source-dialect
  "Select source spelling from the same hardware descriptor used by scheduling.

   A missing optional descriptor still has a conservative family default: Level Zero is the
   explicit Intel path, while generic OpenCL uses portable subgroup spelling."
  [target backend descriptor]
  (or (try
        (some-> descriptor kernel-body-c-dialect)
        (catch Throwable _ nil))
      (case (device/device-type target)
        :ze :opencl-intel
        :ocl :opencl-portable
        :cuda :cuda
        :hip :hip
        (case backend
          :cuda :cuda
          :hip :hip
          nil))))

