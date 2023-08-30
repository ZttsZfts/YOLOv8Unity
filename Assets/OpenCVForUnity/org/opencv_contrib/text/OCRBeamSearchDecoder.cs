#if !UNITY_WSA_10_0

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TextModule
{

    // C++: class OCRBeamSearchDecoder
    /**
     * OCRBeamSearchDecoder class provides an interface for OCR using Beam Search algorithm.
     *
     * <b>Note:</b>
     * <ul>
     *   <li>
     *       (C++) An example on using OCRBeamSearchDecoder recognition combined with scene text detection can
     *         be found at the demo sample:
     *         &lt;https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/word_recognition.cpp&gt;
     *   </li>
     * </ul>
     */

    public class OCRBeamSearchDecoder : BaseOCR
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
                        text_OCRBeamSearchDecoder_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal OCRBeamSearchDecoder(IntPtr addr) : base(addr) { }

        // internal usage only
        public static new OCRBeamSearchDecoder __fromPtr__(IntPtr addr) { return new OCRBeamSearchDecoder(addr); }

        //
        // C++: static Ptr_OCRBeamSearchDecoder cv::text::OCRBeamSearchDecoder::create(Ptr_OCRBeamSearchDecoder_ClassifierCallback classifier, String vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode = OCR_DECODER_VITERBI, int beam_size = 500)
        //

        // Unknown type 'Ptr_OCRBeamSearchDecoder_ClassifierCallback' (I), skipping the function


        //
        // C++: static Ptr_OCRBeamSearchDecoder cv::text::OCRBeamSearchDecoder::create(String filename, String vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode = OCR_DECODER_VITERBI, int beam_size = 500)
        //

        /**
         * Creates an instance of the OCRBeamSearchDecoder class. Initializes HMMDecoder from the specified path.
         *
         *     
         * param filename automatically generated
         * param vocabulary automatically generated
         * param transition_probabilities_table automatically generated
         * param emission_probabilities_table automatically generated
         * param mode automatically generated
         * param beam_size automatically generated
         * return automatically generated
         */
        public static OCRBeamSearchDecoder create(string filename, string vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode, int beam_size)
        {
            if (transition_probabilities_table != null) transition_probabilities_table.ThrowIfDisposed();
            if (emission_probabilities_table != null) emission_probabilities_table.ThrowIfDisposed();

            return OCRBeamSearchDecoder.__fromPtr__(text_OCRBeamSearchDecoder_create_10(filename, vocabulary, transition_probabilities_table.nativeObj, emission_probabilities_table.nativeObj, mode, beam_size));


        }

        /**
         * Creates an instance of the OCRBeamSearchDecoder class. Initializes HMMDecoder from the specified path.
         *
         *     
         * param filename automatically generated
         * param vocabulary automatically generated
         * param transition_probabilities_table automatically generated
         * param emission_probabilities_table automatically generated
         * param mode automatically generated
         * return automatically generated
         */
        public static OCRBeamSearchDecoder create(string filename, string vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode)
        {
            if (transition_probabilities_table != null) transition_probabilities_table.ThrowIfDisposed();
            if (emission_probabilities_table != null) emission_probabilities_table.ThrowIfDisposed();

            return OCRBeamSearchDecoder.__fromPtr__(text_OCRBeamSearchDecoder_create_11(filename, vocabulary, transition_probabilities_table.nativeObj, emission_probabilities_table.nativeObj, mode));


        }

        /**
         * Creates an instance of the OCRBeamSearchDecoder class. Initializes HMMDecoder from the specified path.
         *
         *     
         * param filename automatically generated
         * param vocabulary automatically generated
         * param transition_probabilities_table automatically generated
         * param emission_probabilities_table automatically generated
         * return automatically generated
         */
        public static OCRBeamSearchDecoder create(string filename, string vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table)
        {
            if (transition_probabilities_table != null) transition_probabilities_table.ThrowIfDisposed();
            if (emission_probabilities_table != null) emission_probabilities_table.ThrowIfDisposed();

            return OCRBeamSearchDecoder.__fromPtr__(text_OCRBeamSearchDecoder_create_12(filename, vocabulary, transition_probabilities_table.nativeObj, emission_probabilities_table.nativeObj));


        }


        //
        // C++:  String cv::text::OCRBeamSearchDecoder::run(Mat image, Mat mask, int min_confidence, int component_level = 0)
        //

        public string run(Mat image, Mat mask, int min_confidence, int component_level)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();
            if (mask != null) mask.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRBeamSearchDecoder_run_10(nativeObj, image.nativeObj, mask.nativeObj, min_confidence, component_level));

            return retVal;
        }

        public string run(Mat image, Mat mask, int min_confidence)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();
            if (mask != null) mask.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRBeamSearchDecoder_run_11(nativeObj, image.nativeObj, mask.nativeObj, min_confidence));

            return retVal;
        }


        //
        // C++:  String cv::text::OCRBeamSearchDecoder::run(Mat image, int min_confidence, int component_level = 0)
        //

        /**
         * Recognize text using Beam Search.
         *
         *     Takes image on input and returns recognized text in the output_text parameter. Optionally
         *     provides also the Rects for individual text elements found (e.g. words), and the list of those
         *     text elements with their confidence values.
         *
         *     param image Input binary image CV_8UC1 with a single text line (or word).
         *
         *
         *     text elements found (e.g. words).
         *
         *     recognition of individual text elements found (e.g. words).
         *
         *     for the recognition of individual text elements found (e.g. words).
         *
         *     param component_level Only OCR_LEVEL_WORD is supported.
         * param min_confidence automatically generated
         * return automatically generated
         */
        public string run(Mat image, int min_confidence, int component_level)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRBeamSearchDecoder_run_12(nativeObj, image.nativeObj, min_confidence, component_level));

            return retVal;
        }

        /**
         * Recognize text using Beam Search.
         *
         *     Takes image on input and returns recognized text in the output_text parameter. Optionally
         *     provides also the Rects for individual text elements found (e.g. words), and the list of those
         *     text elements with their confidence values.
         *
         *     param image Input binary image CV_8UC1 with a single text line (or word).
         *
         *
         *     text elements found (e.g. words).
         *
         *     recognition of individual text elements found (e.g. words).
         *
         *     for the recognition of individual text elements found (e.g. words).
         *
         * param min_confidence automatically generated
         * return automatically generated
         */
        public string run(Mat image, int min_confidence)
        {
            ThrowIfDisposed();
            if (image != null) image.ThrowIfDisposed();

            string retVal = Marshal.PtrToStringAnsi(text_OCRBeamSearchDecoder_run_13(nativeObj, image.nativeObj, min_confidence));

            return retVal;
        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++: static Ptr_OCRBeamSearchDecoder cv::text::OCRBeamSearchDecoder::create(String filename, String vocabulary, Mat transition_probabilities_table, Mat emission_probabilities_table, int mode = OCR_DECODER_VITERBI, int beam_size = 500)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_create_10(string filename, string vocabulary, IntPtr transition_probabilities_table_nativeObj, IntPtr emission_probabilities_table_nativeObj, int mode, int beam_size);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_create_11(string filename, string vocabulary, IntPtr transition_probabilities_table_nativeObj, IntPtr emission_probabilities_table_nativeObj, int mode);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_create_12(string filename, string vocabulary, IntPtr transition_probabilities_table_nativeObj, IntPtr emission_probabilities_table_nativeObj);

        // C++:  String cv::text::OCRBeamSearchDecoder::run(Mat image, Mat mask, int min_confidence, int component_level = 0)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_run_10(IntPtr nativeObj, IntPtr image_nativeObj, IntPtr mask_nativeObj, int min_confidence, int component_level);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_run_11(IntPtr nativeObj, IntPtr image_nativeObj, IntPtr mask_nativeObj, int min_confidence);

        // C++:  String cv::text::OCRBeamSearchDecoder::run(Mat image, int min_confidence, int component_level = 0)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_run_12(IntPtr nativeObj, IntPtr image_nativeObj, int min_confidence, int component_level);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_OCRBeamSearchDecoder_run_13(IntPtr nativeObj, IntPtr image_nativeObj, int min_confidence);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void text_OCRBeamSearchDecoder_delete(IntPtr nativeObj);

    }
}
#endif