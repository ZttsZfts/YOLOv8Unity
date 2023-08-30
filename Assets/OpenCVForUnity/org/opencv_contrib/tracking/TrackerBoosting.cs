
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TrackingModule
{

    // C++: class TrackerBoosting
    /**
     * the Boosting tracker
     *
     * This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.
     * The classifier uses the surrounding background as negative examples in update step to avoid the
     * drifting problem. The implementation is based on CITE: OLB .
     */

    public class TrackerBoosting : Tracker
    {

        protected override void Dispose(bool disposing)
        {

            try
            {
                if (disposing)
                {
                }
                if (IsEnabledDispose)
                {
                    if (nativeObj != IntPtr.Zero)
                        tracking_TrackerBoosting_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal TrackerBoosting(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new TrackerBoosting __fromPtr__(IntPtr addr) { return new TrackerBoosting(addr); }

        //
        // C++: static Ptr_TrackerBoosting cv::TrackerBoosting::create()
        //

        /**
         * Constructor
         * return automatically generated
         */
        public static TrackerBoosting create()
        {


            return TrackerBoosting.__fromPtr__(tracking_TrackerBoosting_create_10());


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_TrackerBoosting cv::TrackerBoosting::create()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_TrackerBoosting_create_10();

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void tracking_TrackerBoosting_delete(IntPtr nativeObj);

    }
}
