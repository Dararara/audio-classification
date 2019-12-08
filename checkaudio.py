from model import RNN
import scipy.io.wavfile as wa
from python_speech_features import mfcc
from arguments import *
import torch
def checkAudio(path):
    rnn = RNN(input_size, hidden_size, num_layers, num_classes)
    
    sample_rate,waveform = wa.read(path)

    mfcc_feature = mfcc(waveform, sample_rate, nfft= 1256)
    
    test_data = torch.Tensor(mfcc_feature)
                
    test_data = test_data.type(torch.float32)
    
    #test_data, test_labels = test_data, test_labels

    test_pred = rnn(test_data.view(-1, 1,13))
           
    test_pred = test_pred[0]
            
    prob = torch.nn.functional.softmax(test_pred)
            
    pre_cls = torch.argmax(prob)

    if pre_cls == torch.tensor(1):
        answer = '望门投止思张俭'
    elif pre_cls == torch.tensor(2):
        answer = '忍死须臾待杜根'
    elif pre_cls == torch.tensor(3):
        answer = '我自横刀向天笑'
    else:
        answer = '去留肝胆两昆仑'
    return answer
