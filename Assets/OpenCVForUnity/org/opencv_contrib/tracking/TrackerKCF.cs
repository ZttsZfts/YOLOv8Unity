
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TrackingModule
{

    // C++: class TrackerKCF
    /**
     * the KCF (Kernelized Correlation Filter) tracker
     *
     * KCF is a novel tracking framework that utilizes properties of circulant matrix to enhance the processing speed.
     * This tracking method is an implementation of CITE: KCF_ECCV which is extended to KCF with color-names features (CITE: KCF_CN).
     * The original paper of KCF is available at &lt;http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf&gt;
     * as well as the matlab implementation. For more information about KCF with color-names features, please refer to
     * &lt;http://www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/index.html&gt;.
     */

    public class TrackerKCF : Tracker
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
                        tracking_TrackerKCF_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal TrackerKCF(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new TrackerKCF __fromPtr__(IntPtr addr) { return new TrackerKCF(addr); }

        // C++: enum MODE
        public const int GRAY = (1 << 0);
        public const int CN = (1 << 1);
        public const int CUSTOM = (1 << 2);
        //
        // C++: static Ptr_TrackerKCF cv::TrackerKCF::create()
        //

        /**
         * Constructor
         * return automatically generated
         */
        public static TrackerKCF create()
        {


            return TrackerKCF.__fromPtr__(tracking_TrackerKCF_create_10());


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_TrackerKCF cv::TrackerKCF::create()
        [DllImport(LIBNAME)]
        private static extern IntPtr tracking_TrackerKCF_create_10();

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void tracking_TrackerKCF_delete(IntPtr nativeObj);

    }
}
