#if !UNITY_WEBGL

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.Structured_lightModule
{
    // C++: class Params
    /**
     * Parameters of SinusoidalPattern constructor
     * width Projector's width.
     * height Projector's height.
     * nbrOfPeriods Number of period along the patterns direction.
     * shiftValue Phase shift between two consecutive patterns.
     * methodId Allow to choose between FTP, PSP and FAPS.
     * nbrOfPixelsBetweenMarkers Number of pixels between two consecutive markers on the same row.
     * setMarkers Allow to set markers on the patterns.
     * markersLocation vector used to store markers location on the patterns.
     */

    public class Params : DisposableOpenCVObject
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
                        structured_1light_Params_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal Params(IntPtr addr) : base(addr) { }


        public IntPtr getNativeObjAddr() { return nativeObj; }

        // internal usage only
        public static Params __fromPtr__(IntPtr addr) { return new Params(addr); }

        //
        // C++:   cv::structured_light::SinusoidalPattern::Params::Params()
        //

        public Params()
        {


            nativeObj = structured_1light_Params_Params_10();


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:   cv::structured_light::SinusoidalPattern::Params::Params()
        [DllImport(LIBNAME)]
        private static extern IntPtr structured_1light_Params_Params_10();

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void structured_1light_Params_delete(IntPtr nativeObj);

    }
}
#endif