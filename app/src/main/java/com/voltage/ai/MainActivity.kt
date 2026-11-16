package com.voltage.ai

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import okhttp3.Dns
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.InetAddress
import java.net.URL
import java.net.UnknownHostException
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    companion object {
        init { System.loadLibrary("volt_jni") }
        const val CATALOG_URL = "https://raw.githubusercontent.com/lMrVoltl/volt-models/main/models.json"
    }

    private lateinit var webView: WebView
    private var fileUploadCallback: ValueCallback<Array<Uri>>? = null

    private val fileChooserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val uris = WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data)
        fileUploadCallback?.onReceiveValue(uris)
        fileUploadCallback = null
    }

    private val smartDns = object : Dns {
        override fun lookup(hostname: String): List<InetAddress> {
            return try {
                Dns.SYSTEM.lookup(hostname)
            } catch (e: UnknownHostException) {
                val res = URL("https://8.8.8.8/resolve?name=$hostname&type=A").readText()
                val answers = JSONObject(res).optJSONArray("Answer") ?: throw e
                val ips = mutableListOf<InetAddress>()
                for (i in 0 until answers.length()) {
                    val item = answers.getJSONObject(i)
                    if (item.optInt("type") == 1) ips.add(InetAddress.getByName(item.getString("data")))
                }
                if (ips.isEmpty()) throw e else ips
            }
        }
    }

    private val client = OkHttpClient.Builder()
        .dns(smartDns)
        .followRedirects(true)
        .followSslRedirects(true)
        .connectTimeout(25, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    external fun nativeLoadModel(path: String, mmprojPath: String, threads: Int, nCtx: Int): Boolean
    external fun nativeGenerate(prompt: String, sysPrompt: String, rgbBytes: ByteArray?, imgW: Int, imgH: Int, temp: Float, topP: Float, maxTokens: Int)
    external fun nativeStop()

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            webViewClient = WebViewClient()
            webChromeClient = object : WebChromeClient() {
                override fun onShowFileChooser(
                    view: WebView?,
                    filePathCallback: ValueCallback<Array<Uri>>?,
                    fileChooserParams: FileChooserParams?
                ): Boolean {
                    fileUploadCallback?.onReceiveValue(null)
                    fileUploadCallback = filePathCallback
                    val intent = fileChooserParams?.createIntent() ?: return false
                    fileChooserLauncher.launch(intent)
                    return true
                }
            }
            addJavascriptInterface(VoltBridge(), "VoltAndroid")
            loadUrl("file:///android_asset/index.html")
        }
        setContentView(webView)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName")))
        }
    }

    private fun modelsDir(): File {
        val appDir = File(getExternalFilesDir(null) ?: filesDir, "VoltModels")
        if (!appDir.exists()) appDir.mkdirs()
        return appDir
    }

    private fun resolveModelFile(fileName: String): File {
        val appFile = File(modelsDir(), fileName)
        if (appFile.exists() && appFile.length() > 10_000_000L) return appFile
        val pubDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "VoltModels")
        val pubFile = File(pubDir, fileName)
        if (pubFile.exists() && pubFile.length() > 10_000_000L) return pubFile
        return appFile
    }

    private fun downloadSingleFile(urlStr: String, target: File, progressStart: Int, progressSpan: Int, id: String) {
        val tmp = File(target.parentFile, target.name + ".part")
        val req = Request.Builder().url(urlStr).build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw Exception("HTTP ${resp.code}")
            val body = resp.body ?: throw Exception("Нет данных")
            val total = body.contentLength()
            var downloaded = 0L
            var lastPct = -1

            body.byteStream().use { input ->
                FileOutputStream(tmp).use { output ->
                    val buf = ByteArray(65536)
                    while (true) {
                        val r = input.read(buf)
                        if (r <= 0) break
                        output.write(buf, 0, r)
                        downloaded += r
                        if (total > 0) {
                            val pct = progressStart + ((downloaded * progressSpan) / total).toInt()
                            if (pct != lastPct) {
                                lastPct = pct
                                runOnUiThread {
                                    webView.evaluateJavascript("window.onDownloadProgress('$id', $pct)", null)
                                }
                            }
                        }
                    }
                }
            }
        }
        if (target.exists()) target.delete()
        tmp.renameTo(target)
    }

    fun onNativeToken(token: String) {
        val escaped = JSONObject.quote(token)
        runOnUiThread { webView.evaluateJavascript("window.onVoltToken($escaped)", null) }
    }

    fun onNativeStats(tokens: Int, tps: Double, totalSec: Double, visionState: Int, visionErr: String) {
        runOnUiThread {
            val qErr = JSONObject.quote(visionErr)
            webView.evaluateJavascript("window.onVoltStats($tokens, $tps, $totalSec, $visionState, $qErr)", null)
        }
    }

    inner class VoltBridge {
        @JavascriptInterface
        fun getDeviceRamJson(): String {
            val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mi = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            return JSONObject()
                .put("free_mb", mi.availMem / (1024 * 1024))
                .put("total_mb", mi.totalMem / (1024 * 1024))
                .toString()
        }

        @JavascriptInterface
        fun getDeviceHardwareJson(): String {
            val rawSoc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
            val socTitle = when {
                rawSoc.contains("7450", true) -> "Qualcomm Snapdragon 7 Gen 1"
                rawSoc.contains("7325", true) -> "Qualcomm Snapdragon 778G"
                rawSoc.contains("8475", true) -> "Qualcomm Snapdragon 8+ Gen 1"
                rawSoc.contains("8550", true) -> "Qualcomm Snapdragon 8 Gen 2"
                rawSoc.contains("8650", true) -> "Qualcomm Snapdragon 8 Gen 3"
                rawSoc.isNotBlank() && rawSoc != "unknown" -> rawSoc
                else -> Build.MODEL
            }

            val cpuInfo = try { File("/proc/cpuinfo").readText() } catch (_: Exception) { "" }
            val hasDotProd = cpuInfo.contains("asimddp", true)
            val hasFp16 = cpuInfo.contains("fphp", true) || cpuInfo.contains("asimdhp", true)
            val hasI8mm = cpuInfo.contains("i8mm", true)

            val simdList = mutableListOf("ARM64 NEON")
            if (hasDotProd) simdList.add("DotProd")
            if (hasFp16) simdList.add("FP16")
            if (hasI8mm) simdList.add("I8MM")

            val cores = Runtime.getRuntime().availableProcessors()
            val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mi = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            val totalGb = Math.round((mi.totalMem / (1024.0 * 1024.0 * 1024.0)) * 10.0) / 10.0

            val supported = JSONArray().apply {
                put(JSONObject().put("tier", "Volt Ion v2.5 • 1.5B Fast").put("quant", "Qwen 2.5 1.5B • Текст и файлы").put("status", "Максимальная скорость"))
                put(JSONObject().put("tier", "Volt Neutron v2.5 • 1.5B Uncensored").put("quant", "Qwen 2.5 1.5B Abliterated • RP / Без фильтров").put("status", "Свободный диалог"))
                put(JSONObject().put("tier", "Volt Photon v2.0 • 2B Vision").put("quant", "Qwen2-VL 2B + mmproj FP16").put("status", "Субфлагман (фото + текст)"))
                if (totalGb >= 7.5) {
                    put(JSONObject().put("tier", "Volt Plasma v2.5 • 3B Vision").put("quant", "Qwen2.5-VL 3B + mmproj Q8_0").put("status", if (hasDotProd) "Флагман (ускорение DotProd)" else "Флагман"))
                }
            }

            return JSONObject()
                .put("device", "${Build.MANUFACTURER.uppercase()} ${Build.MODEL}")
                .put("soc", socTitle)
                .put("socCode", rawSoc)
                .put("cores", cores)
                .put("totalGb", totalGb)
                .put("simd", simdList.joinToString(" • "))
                .put("supported", supported)
                .toString()
        }

        @JavascriptInterface
        fun recognizeImageOcr(base64Data: String) {
            try {
                val clean = base64Data.substringAfter(",")
                val bytes = Base64.decode(clean, Base64.DEFAULT)
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                val image = InputImage.fromBitmap(bmp, 0)
                val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
                recognizer.process(image)
                    .addOnSuccessListener { visionText ->
                        val q = JSONObject.quote(visionText.text)
                        webView.evaluateJavascript("window.onOcrDone($q)", null)
                    }
                    .addOnFailureListener {
                        webView.evaluateJavascript("window.onOcrDone('')", null)
                    }
            } catch (_: Exception) {
                runOnUiThread { webView.evaluateJavascript("window.onOcrDone('')", null) }
            }
        }

        @JavascriptInterface
        fun fetchCatalog() {
            thread {
                try {
                    val req = Request.Builder().url(CATALOG_URL).build()
                    val raw = client.newCall(req).execute().use { resp ->
                        if (!resp.isSuccessful) throw Exception("HTTP ${resp.code}")
                        resp.body?.string() ?: throw Exception("Пустой ответ")
                    }
                    val json = JSONObject(raw)
                    val models = json.getJSONArray("models")
                    val dir = modelsDir()
                    for (i in 0 until models.length()) {
                        val m = models.getJSONObject(i)
                        val id = m.getString("id")
                        val file = resolveModelFile("$id.gguf")
                        val hasMmproj = m.optString("mmproj_url", "").isNotEmpty()
                        val mmFile = resolveModelFile("$id-mmproj.gguf")
                        val ready = file.exists() && file.length() > 10_000_000 &&
                            (!hasMmproj || (mmFile.exists() && mmFile.length() > 10_000_000))
                        m.put("downloaded", ready)
                    }
                    val quoted = JSONObject.quote(json.toString())
                    runOnUiThread { webView.evaluateJavascript("window.onCatalogLoaded($quoted)", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка сети")
                    runOnUiThread { webView.evaluateJavascript("window.onCatalogError($err)", null) }
                }
            }
        }

        @JavascriptInterface
        fun downloadModel(id: String, urlStr: String, mmprojUrl: String) {
            thread {
                try {
                    val dir = modelsDir()
                    val target = File(dir, "$id.gguf")
                    if (mmprojUrl.isNotEmpty()) {
                        downloadSingleFile(urlStr, target, 0, 65, id)
                        val mmTarget = File(dir, "$id-mmproj.gguf")
                        downloadSingleFile(mmprojUrl, mmTarget, 65, 35, id)
                    } else {
                        downloadSingleFile(urlStr, target, 0, 100, id)
                    }
                    runOnUiThread { webView.evaluateJavascript("window.onDownloadDone('$id')", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка загрузки")
                    runOnUiThread { webView.evaluateJavascript("window.onDownloadError('$id', $err)", null) }
                }
            }
        }

        @JavascriptInterface
        fun loadModel(id: String, threads: Int, nCtx: Int) {
            thread {
                val file = resolveModelFile("$id.gguf")
                val mmFile = resolveModelFile("$id-mmproj.gguf")
                val mmPath = if (mmFile.exists() && mmFile.length() > 10_000_000) mmFile.absolutePath else ""
                val ok = file.exists() && nativeLoadModel(file.absolutePath, mmPath, threads, nCtx)
                runOnUiThread { webView.evaluateJavascript("window.onModelLoaded('$id', $ok)", null) }
            }
        }

        @JavascriptInterface
        fun sendPrompt(formattedHistory: String, sysPrompt: String, imgBase64: String, temp: Float, topP: Float, maxTokens: Int) {
            thread {
                var rgbBytes: ByteArray? = null
                var w = 0
                var h = 0
                if (imgBase64.isNotEmpty()) {
                    try {
                        val clean = imgBase64.substringAfter(",")
                        val bytes = Base64.decode(clean, Base64.DEFAULT)
                        val rawBmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        val maxSide = 336
                        val scale = minOf(maxSide.toFloat() / rawBmp.width, maxSide.toFloat() / rawBmp.height, 1.0f)
                        val targetW = maxOf(28, (((rawBmp.width * scale).toInt() + 14) / 28) * 28)
                        val targetH = maxOf(28, (((rawBmp.height * scale).toInt() + 14) / 28) * 28)
                        val scaled = Bitmap.createScaledBitmap(rawBmp, targetW, targetH, true)
                        w = scaled.width
                        h = scaled.height
                        val pixels = IntArray(w * h)
                        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
                        val rgb = ByteArray(w * h * 3)
                        for (i in pixels.indices) {
                            val px = pixels[i]
                            rgb[i * 3] = ((px shr 16) and 0xFF).toByte()
                            rgb[i * 3 + 1] = ((px shr 8) and 0xFF).toByte()
                            rgb[i * 3 + 2] = (px and 0xFF).toByte()
                        }
                        rgbBytes = rgb
                    } catch (_: Exception) {}
                }
                nativeGenerate(formattedHistory, sysPrompt, rgbBytes, w, h, temp, topP, maxTokens)
                runOnUiThread { webView.evaluateJavascript("window.onVoltDone()", null) }
            }
        }

        @JavascriptInterface
        fun stopGeneration() { nativeStop() }
    }
}
