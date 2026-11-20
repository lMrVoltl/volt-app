package com.voltage.ai

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Bitmap
import android.graphics.Path
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.util.Base64
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class VoltAgentService : AccessibilityService() {

    companion object {
        @Volatile
        var instance: VoltAgentService? = null

        private val lastNodes = mutableListOf<AccessibilityNodeInfo>()

        fun isEnabled(): Boolean = instance != null

        fun dumpScreenTree(): String {
            val svc = instance ?: return """{"error":"Спец. возможности отключены. Включи Volt AI в настройках Accessibility."}"""
            val root = svc.rootInActiveWindow ?: return """{"error":"Активное окно недоступно"}"""
            lastNodes.clear()
            val arr = JSONArray()
            val pkg = root.packageName?.toString() ?: "unknown"

            fun walk(node: AccessibilityNodeInfo?) {
                if (node == null || arr.length() >= 70) return
                val txt = node.text?.toString()?.trim().orEmpty()
                val desc = node.contentDescription?.toString()?.trim().orEmpty()
                val click = node.isClickable
                val edit = node.isEditable

                if (txt.isNotEmpty() || desc.isNotEmpty() || click || edit) {
                    val rect = Rect()
                    node.getBoundsInScreen(rect)
                    if (rect.width() > 0 && rect.height() > 0) {
                        val id = lastNodes.size
                        lastNodes.add(node)
                        val obj = JSONObject().put("id", id)
                        if (txt.isNotEmpty()) obj.put("text", txt.take(90))
                        if (desc.isNotEmpty() && desc != txt) obj.put("desc", desc.take(70))
                        if (click) obj.put("clickable", true)
                        if (edit) obj.put("editable", true)
                        obj.put("xy", "${rect.centerX()},${rect.centerY()}")
                        arr.put(obj)
                    }
                }
                for (i in 0 until node.childCount) {
                    walk(node.getChild(i))
                }
            }

            walk(root)
            return JSONObject()
                .put("package", pkg)
                .put("elements_count", arr.length())
                .put("elements", arr)
                .toString()
        }

        fun clickById(id: Int): Boolean {
            val node = lastNodes.getOrNull(id) ?: return false
            var cur: AccessibilityNodeInfo? = node
            while (cur != null) {
                if (cur.isClickable && cur.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
                cur = cur.parent
            }
            val r = Rect()
            node.getBoundsInScreen(r)
            return tapAt(r.centerX().toFloat(), r.centerY().toFloat())
        }

        fun clickByText(query: String): Boolean {
            val svc = instance ?: return false
            val root = svc.rootInActiveWindow ?: return false
            val q = query.trim().lowercase()
            if (q.isEmpty()) return false

            fun search(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
                if (node == null) return null
                val t = node.text?.toString()?.lowercase().orEmpty()
                val d = node.contentDescription?.toString()?.lowercase().orEmpty()
                if (t.contains(q) || d.contains(q)) return node
                for (i in 0 until node.childCount) {
                    val found = search(node.getChild(i))
                    if (found != null) return found
                }
                return null
            }

            val target = search(root) ?: return false
            var cur: AccessibilityNodeInfo? = target
            while (cur != null) {
                if (cur.isClickable && cur.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
                cur = cur.parent
            }
            val r = Rect()
            target.getBoundsInScreen(r)
            return tapAt(r.centerX().toFloat(), r.centerY().toFloat())
        }

        fun tapAt(x: Float, y: Float): Boolean {
            val svc = instance ?: return false
            val path = Path().apply { moveTo(x, y) }
            val stroke = GestureDescription.StrokeDescription(path, 0, 60)
            return svc.dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
        }

        fun swipe(x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long = 320): Boolean {
            val svc = instance ?: return false
            val path = Path().apply {
                moveTo(x1, y1)
                lineTo(x2, y2)
            }
            val stroke = GestureDescription.StrokeDescription(path, 0, durationMs.coerceIn(60, 2000))
            return svc.dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
        }

        fun scrollDown(): Boolean {
            val dm = instance?.resources?.displayMetrics ?: return false
            val cx = dm.widthPixels / 2f
            return swipe(cx, dm.heightPixels * 0.75f, cx, dm.heightPixels * 0.28f, 350)
        }

        fun scrollUp(): Boolean {
            val dm = instance?.resources?.displayMetrics ?: return false
            val cx = dm.widthPixels / 2f
            return swipe(cx, dm.heightPixels * 0.30f, cx, dm.heightPixels * 0.78f, 350)
        }

        fun inputText(text: String, nodeId: Int = -1): Boolean {
            val svc = instance ?: return false
            val root = svc.rootInActiveWindow ?: return false
            val target = (if (nodeId >= 0) lastNodes.getOrNull(nodeId) else null)
                ?: root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                ?: findEditable(root)
                ?: return false
            target.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
            val args = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
            }
            return target.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        }

        private fun findEditable(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
            if (node == null) return null
            if (node.isEditable) return node
            for (i in 0 until node.childCount) {
                val res = findEditable(node.getChild(i))
                if (res != null) return res
            }
            return null
        }

        fun goBack(): Boolean = instance?.performGlobalAction(GLOBAL_ACTION_BACK) ?: false
        fun goHome(): Boolean = instance?.performGlobalAction(GLOBAL_ACTION_HOME) ?: false
        fun openNotifications(): Boolean = instance?.performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS) ?: false

        fun captureScreenDataUrl(): String {
            val svc = instance ?: return ""
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return ""
            var dataUrl = ""
            val latch = CountDownLatch(1)
            svc.takeScreenshot(Display.DEFAULT_DISPLAY, svc.mainExecutor, object : TakeScreenshotCallback {
                override fun onSuccess(screenshot: ScreenshotResult) {
                    try {
                        val hw = Bitmap.wrapHardwareBuffer(screenshot.hardwareBuffer, screenshot.colorSpace)
                        val soft = hw?.copy(Bitmap.Config.ARGB_8888, false)
                        screenshot.hardwareBuffer.close()
                        if (soft != null) {
                            val maxSide = 448
                            val sc = minOf(maxSide.toFloat() / soft.width, maxSide.toFloat() / soft.height, 1.0f)
                            val w = maxOf(28, (((soft.width * sc).toInt() + 14) / 28) * 28)
                            val h = maxOf(28, (((soft.height * sc).toInt() + 14) / 28) * 28)
                            val scaled = Bitmap.createScaledBitmap(soft, w, h, true)
                            val baos = ByteArrayOutputStream()
                            scaled.compress(Bitmap.CompressFormat.JPEG, 75, baos)
                            val b64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
                            dataUrl = "data:image/jpeg;base64,$b64"
                        }
                    } catch (_: Exception) {
                    } finally {
                        latch.countDown()
                    }
                }
                override fun onFailure(errorCode: Int) {
                    latch.countDown()
                }
            })
            latch.await(3, TimeUnit.SECONDS)
            return dataUrl
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {}

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }
}
