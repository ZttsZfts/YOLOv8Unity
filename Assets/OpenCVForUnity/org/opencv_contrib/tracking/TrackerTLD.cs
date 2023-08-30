
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TrackingModule
{

    // C++: class TrackerTLD
    /**
     * the TLD (Tracking, learning and detection) tracker
     *
     * TLD is a novel tracking framework that explicitly decomposes the long-term tracking task into
     * tracking, learning and detection.
     *
     * The tracker follows the object from frame to frame. The detector localizes all appearances that
     * have been observed so far and corrects the tracker if necessary. The learning estimates detector's
     * errors and updates it to avoid these errors in the future. The implementation is based on CITE: TLD .
     *
     * The Median Flow algorithm (see cv::TrackerMedianFlow) was chosen as a tracking component in this
     * implementation, following authors. The tracker is supposed to be able to handle rapid motions, partial
     * occlusions, object absence etc.
     */

    public class TrackerTLD : Tracker
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
                        tracking_TrackerTLD_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal TrackerTLD(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new TrackerTLD __fromPtr__(IntPtr addr) { return new TrackerTLD(addr); }

        //
        // C++: static Ptr_TrackerTLD cv::TrackerTLD::create()
        //

        /**
         * Constructor
         * return automatically generated
         */
        public static TrackerTLD create()
        {


            return TrackerTLD.__fromPtr__(tracking_TrackerTLD_create_10());


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_TrackerTLD cv::TrackerTLD::create()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_TrackerTLD_create_10();

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void tracking_TrackerTLD_delete(IntPtr nativeObj);

    }
}
