
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TrackingModule
{

    // C++: class TrackerMOSSE
    /**
     * the MOSSE (Minimum Output Sum of Squared %Error) tracker
     *
     * The implementation is based on CITE: MOSSE Visual Object Tracking using Adaptive Correlation Filters
     * <b>Note:</b> this tracker works with grayscale images, if passed bgr ones, they will get converted internally.
     */

    public class TrackerMOSSE : Tracker
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
                        tracking_TrackerMOSSE_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal TrackerMOSSE(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new TrackerMOSSE __fromPtr__(IntPtr addr) { return new TrackerMOSSE(addr); }

        //
        // C++: static Ptr_TrackerMOSSE cv::TrackerMOSSE::create()
        //

        /**
         * Constructor
         * return automatically generated
         */
        public static TrackerMOSSE create()
        {


            return TrackerMOSSE.__fromPtr__(tracking_TrackerMOSSE_create_10());


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_TrackerMOSSE cv::TrackerMOSSE::create()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_TrackerMOSSE_create_10();

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void tracking_TrackerMOSSE_delete(IntPtr nativeObj);

    }
}
