package com.voltage.ai

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.os.Bundle
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import okhttp3.Dns
import okhttp3.OkHttpClient
import okhttp3.Request
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
                    if (item.optInt("type") == 1) {
                        ips.add(InetAddress.getByName(item.getString("data")))
                    }
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

    external fun nativeLoadModel(path: String, threads: Int): Boolean
    external fun nativeGenerate(prompt: String)
    external fun nativeStop()

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            webViewClient = WebViewClient()
            addJavascriptInterface(VoltBridge(), "VoltAndroid")
            loadUrl("file:///android_asset/index.html")
        }
        setContentView(webView)
    }

    fun onNativeToken(token: String) {
        val escaped = JSONObject.quote(token)
        runOnUiThread { webView.evaluateJavascript("window.onVoltToken($escaped)", null) }
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
                    val dir = getExternalFilesDir("models")!!
                    for (i in 0 until models.length()) {
                        val m = models.getJSONObject(i)
                        val file = File(dir, m.getString("id") + ".gguf")
                        m.put("downloaded", file.exists() && file.length() > 10_000_000)
                    }
                    val quoted = JSONObject.quote(json.toString())
                    runOnUiThread {
                        webView.evaluateJavascript("window.onCatalogLoaded($quoted)", null)
                    }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка сети")
                    runOnUiThread {
                        webView.evaluateJavascript("window.onCatalogError($err)", null)
                    }
                }
            }
        }

        @JavascriptInterface
        fun downloadModel(id: String, urlStr: String) {
            thread {
                try {
                    val dir = getExternalFilesDir("models")!!
                    if (!dir.exists()) dir.mkdirs()
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
                    runOnUiThread {
                        webView.evaluateJavascript("window.onDownloadError('$id', $err)", null)
                    }
                }
            }
        }

        @JavascriptInterface
        fun loadModel(id: String) {
            thread {
                val file = File(getExternalFilesDir("models"), "$id.gguf")
                val ok = file.exists() && nativeLoadModel(file.absolutePath, 4)
                runOnUiThread { webView.evaluateJavascript("window.onModelLoaded('$id', $ok)", null) }
            }
        }

        @JavascriptInterface
        fun sendPrompt(prompt: String) {
            thread {
                nativeGenerate(prompt)
                runOnUiThread { webView.evaluateJavascript("window.onVoltDone()", null) }
            }
        }

        @JavascriptInterface
        fun stopGeneration() { nativeStop() }
    }
}
