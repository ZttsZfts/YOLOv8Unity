
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TrackingModule
{

    // C++: class TrackerMedianFlow
    /**
     * the Median Flow tracker
     *
     * Implementation of a paper CITE: MedianFlow .
     *
     * The tracker is suitable for very smooth and predictable movements when object is visible throughout
     * the whole sequence. It's quite and accurate for this type of problems (in particular, it was shown
     * by authors to outperform MIL). During the implementation period the code at
     * &lt;http://www.aonsquared.co.uk/node/5&gt;, the courtesy of the author Arthur Amarra, was used for the
     * reference purpose.
     */

    public class TrackerMedianFlow : Tracker
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
                        tracking_TrackerMedianFlow_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal TrackerMedianFlow(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new TrackerMedianFlow __fromPtr__(IntPtr addr) { return new TrackerMedianFlow(addr); }

        //
        // C++: static Ptr_TrackerMedianFlow cv::TrackerMedianFlow::create()
        //

        /**
         * Constructor
         * return automatically generated
         */
        public static TrackerMedianFlow create()
        {


            return TrackerMedianFlow.__fromPtr__(tracking_TrackerMedianFlow_create_10());


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_TrackerMedianFlow cv::TrackerMedianFlow::create()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_TrackerMedianFlow_create_10();

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void tracking_TrackerMedianFlow_delete(IntPtr nativeObj);

    }
}
