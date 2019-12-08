import wx 
from checkaudio import checkAudio
from playAudio import play
class MyFileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):
        '''
        fig = self.window.figure
        inaxes = fig.get_axes()[0]
        h_pix = int(fig.get_figheight() * fig.get_dpi()) # fig height in pixels
        message = "%d file(s) dropped at (%d,%d):\n" % (len(filenames), x, y)
        for file in filenames:
            message += file + "\n"
        inaxes.annotate(message, (x, h_pix-y), xycoords='figure pixels')     
        '''
        #print(filenames)
        path = ''.join(filenames)
        #print(path)
        #print(checkAudio(path))
        play(path)
        wx.StaticText(self.window, label = checkAudio(path) , pos = (100,100)) 
        return True
        
if __name__ == '__main__':
    app = wx.App() 
    window = wx.Frame(None, title = "hahahahah", size = (400,300)) 
    panel = wx.Panel(window)    
    label = wx.StaticText(panel, label = "拖入框内判定", pos = (100,100)) 
    win_target = panel
    dt = MyFileDropTarget(win_target)
    win_target.SetDropTarget(dt)
    window.Show(True) 
    app.MainLoop()