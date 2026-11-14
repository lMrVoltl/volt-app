package com.voltage.ai

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
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
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()

    external fun nativeLoadModel(path: String, threads: Int, nCtx: Int): Boolean
    external fun nativeGenerate(prompt: String, sysPrompt: String, temp: Float, topP: Float, maxTokens: Int)
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
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "VoltModels")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun onNativeToken(token: String) {
        val escaped = JSONObject.quote(token)
        runOnUiThread { webView.evaluateJavascript("window.onVoltToken($escaped)", null) }
    }

    fun onNativeStats(tokens: Int, tps: Double, totalSec: Double) {
        runOnUiThread {
            webView.evaluateJavascript("window.onVoltStats($tokens, $tps, $totalSec)", null)
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
                put(JSONObject().put("tier", "0.5B • Volt Photon").put("quant", "Q4_K_M / Q5_K_M / Q8_0").put("status", "Максимальная скорость"))
                put(JSONObject().put("tier", "1.5B – 2B • Volt Plasma").put("quant", "Q4_K_M / Q5_K_M").put("status", "Оптимально"))
                if (totalGb >= 7.5) {
                    put(JSONObject().put("tier", "3B – 4B • Heavy").put("quant", "Q4_K_M").put("status", if (hasDotProd) "Поддерживается (ускорение DotProd)" else "Поддерживается"))
                }
                if (totalGb >= 11.0) {
                    put(JSONObject().put("tier", "7B – 8B • Ultra").put("quant", "IQ4_XS / Q4_K_M").put("status", "Запуск при >4.5 ГБ свободной RAM"))
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
                        val file = File(dir, m.getString("id") + ".gguf")
                        m.put("downloaded", file.exists() && file.length() > 10_000_000)
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
        fun downloadModel(id: String, urlStr: String) {
            thread {
                try {
                    val dir = modelsDir()
                    val target = File(dir, "$id.gguf")
                    val tmp = File(dir, "$id.gguf.part")

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
                                        val pct = ((downloaded * 100) / total).toInt()
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
                    tmp.renameTo(target)
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
                val file = File(modelsDir(), "$id.gguf")
                val ok = file.exists() && nativeLoadModel(file.absolutePath, threads, nCtx)
                runOnUiThread { webView.evaluateJavascript("window.onModelLoaded('$id', $ok)", null) }
            }
        }

        @JavascriptInterface
        fun sendPrompt(formattedHistory: String, sysPrompt: String, temp: Float, topP: Float, maxTokens: Int) {
            thread {
                nativeGenerate(formattedHistory, sysPrompt, temp, topP, maxTokens)
                runOnUiThread { webView.evaluateJavascript("window.onVoltDone()", null) }
            }
        }

        @JavascriptInterface
        fun stopGeneration() { nativeStop() }
    }
}
