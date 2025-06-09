import win32com.client
import pythoncom
import threading

# 全局Word实例和锁
word_app = None
word_lock = threading.Lock()

def get_word_app():
    """获取全局Word实例"""
    global word_app
    if word_app is None:
        pythoncom.CoInitialize()
        word_app = win32com.client.Dispatch("Word.Application")
        word_app.Visible = False
        word_app.DisplayAlerts = False
    return word_app

def cleanup_word_app():
    """清理Word实例"""
    global word_app
    if word_app is not None:
        try:
            word_app.Quit()
        except:
            pass
        word_app = None
    pythoncom.CoUninitialize() 