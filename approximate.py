import numpy as np

class Approximate:
    def __init__(self, macs, res, prune, model_type = 'VGG16'):
        self.macs = macs
        self.resolution = res
        self.prune = prune
        self.model_type = model_type
    
    def get_approximates(self):
        return self.prune_accuracy()*self.resolution_accuracy(), self.macs_gpu_fps()

    def macs_gpu_fps(self):
        macs = self.macs/1e9
        if self.model_type == 'VGG16':
            flops_absolute = [61, 138, 246, 384, 553, 752, 983, 1244, 1536]
            gpu_fps_absolute = [60, 37, 22.7, 14, 10, 7.21,5.7, 4.5, 3]
            function = np.polyfit(flops_absolute, gpu_fps_absolute, 5)
            predicted_gpu_fps = np.poly1d(function)
            return predicted_gpu_fps(macs)
        
        if self.model_type == 'MOBILENET':
            flops_absolute = [314130496, 722661248, 1252681984, 1985437504, 2816934464, 3873914752, 5006887936, 5591688880, 6118529680, 6752395600, 7321484128, 7822542400, 8549414896]
            flops_absolute = [i/1e9 for i in flops_absolute]
            gpu_fps_absolute = [60, 60, 60, 60, 60, 60, 60, 57, 53, 48, 44, 42, 38]
            function = np.polyfit(flops_absolute, gpu_fps_absolute, 5)
            predicted_gpu_fps = np.poly1d(function)
            return predicted_gpu_fps(macs)

        if self.model_type == 'INCEPTION':
            flops_absolute = [2847271040, 540343872, 21444719168, 72814356032, 43321379904, 7184373824, 13215394176, 31199832704, 56800586624, 90017655936]
            flops_absolute = [i/1e9 for i in flops_absolute]
            gpu_fps_absolute = [60, 60, 52, 27, 41, 59, 56.5, 47, 35, 24]
            function = np.polyfit(flops_absolute, gpu_fps_absolute, 4)
            predicted_gpu_fps = np.poly1d(function)
            return predicted_gpu_fps(macs)

        if self.model_type == 'RESNET':
            flops_absolute = [4111514624, 9403591168, 16439908352, 25942539776, 36987231232, 50700417536, 65753483264, 83677224448, 102738664448]
            flops_absolute = [i/1e9 for i in flops_absolute]
            gpu_fps_absolute = [60, 60, 60, 60, 60, 45, 37, 28, 23]
            function = np.polyfit(flops_absolute, gpu_fps_absolute, 5)
            predicted_gpu_fps = np.poly1d(function)
            return predicted_gpu_fps(macs)

        if self.model_type == 'DENSENET':
            flops_absolute = [0, 2865672192, 6424335872, 11459616768, 17867804160, 25782857728, 35040568832, 45835395072, 57942629888, 71617228800]
            flops_absolute = [i/1e9 for i in flops_absolute]
            gpu_fps_absolute = [60, 42.31, 41.68, 39.67, 38.57, 36.64, 36, 31.71, 25.68, 21.84]
            function = np.polyfit(flops_absolute, gpu_fps_absolute, 3)
            predicted_gpu_fps = np.poly1d(function)
            return predicted_gpu_fps(macs)

    def prune_accuracy(self):
        if self.model_type == 'VGG16':
            prune_absolute = list(np.arange(0,1.1,0.1))
            accuracy_percentage = [85, 85, 63, 60, 21, 10, 10, 10, 10, 0, 0]
            accuracy_percentage = [i/max(accuracy_percentage) for i in accuracy_percentage]
            function = np.polyfit(prune_absolute, accuracy_percentage, 4)
            predicted_accuracy_percentage = np.poly1d(function)
            return predicted_accuracy_percentage(self.prune)
        
        if self.model_type == 'MOBILENET':
            prune_absolute = [0, 0.2, 0.48, 0.5, 0.75, 0.82, 0.85, 0.9, 0.95]
            accuracy_percentage = [0.706, 0.702, 0.70, 0.695, 0.677, 0.64, .621, 0.618, 0.536]
            accuracy_percentage = [i/max(accuracy_percentage) for i in accuracy_percentage]
            function = np.polyfit(prune_absolute, accuracy_percentage, 3)
            predicted_accuracy_percentage = np.poly1d(function)
            return predicted_accuracy_percentage(self.prune)
        
        if self.model_type == 'INCEPTION':
            prune_absolute = [0,  0.1 , 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.9, 0.95, 1]
            accuracy_percentage = [0.781, 0.781, 0.781, 0.781, 0.781, 0.781, 0.761, 0.746, 0.7, 0.6, 0.52]
            accuracy_percentage = [i/max(accuracy_percentage) for i in accuracy_percentage]
            function = np.polyfit(prune_absolute, accuracy_percentage, 5)
            predicted_accuracy_percentage = np.poly1d(function)
            return predicted_accuracy_percentage(self.prune)

        if self.model_type == 'RESNET':
            prune_absolute = [0, 0.1, 0.2, 0.3, 0.41,  0.5, 0.45, 0.266, 0.422, 0.434, .355, .440, 0.488, 0.3288, 0.435, 0.6, 0.7, 0.8, 0.9, 1]
            accuracy_percentage = [0.78, 0.78, 0.78, 0.762, 0.7461, 0.7261, 0.7450, 0.7600, 0.7505, 0.7182, 0.7348, 0.7490, 0.7250, 0.7553, 0.7120, 0.69, 0.66, 0.62, 0.59, 0]
            accuracy_percentage = [i/max(accuracy_percentage) for i in accuracy_percentage]
            function = np.polyfit(prune_absolute, accuracy_percentage, 4)
            predicted_accuracy_percentage = np.poly1d(function)
            return min(1, predicted_accuracy_percentage(self.prune))

        if self.model_type == 'DENSENET':
            prune_absolute = [0, 105/556, 263/556, 173/556, 716/3630, 2420/3630, 2570/5670, 0.4478, 0.2267, 0.6, 0.36, 1]
            accuracy_percentage = [1, (1-0.0682)/(1-0.0524), (1-0.2699)/(1-0.2509), (1-0.06)/(1-0.0524), (1-0.0512)/(1-0.0434), (1-0.2414)/(1-0.2008), (1-0.2719)/(1-0.2535), 0.9316/0.9411, 0.7219/0.7464, (1-0.2449)/(1-0.2320), (1-0.2372)/(1-0.2320), 0]
            function = np.polyfit(prune_absolute, accuracy_percentage, 4)
            predicted_accuracy_percentage = np.poly1d(function)
            return min(1, predicted_accuracy_percentage(self.prune))

        if self.model_type == 'SHUFFLENET':
            prune_absolute = [0, 105/556, 263/556, 173/556, 716/3630, 2420/3630, 2570/5670, 0.4478, 0.2267, 0.6, 0.36, 1]
            accuracy_percentage = [1, (1-0.0682)/(1-0.0524), (1-0.2699)/(1-0.2509), (1-0.06)/(1-0.0524), (1-0.0512)/(1-0.0434), (1-0.2414)/(1-0.2008), (1-0.2719)/(1-0.2535), 0.9316/0.9411, 0.7219/0.7464, (1-0.2449)/(1-0.2320), (1-0.2372)/(1-0.2320), 0]
            function = np.polyfit(prune_absolute, accuracy_percentage, 4)
            predicted_accuracy_percentage = np.poly1d(function)
            return min(1, predicted_accuracy_percentage(self.prune))


    def resolution_accuracy(self):
        resolution = 1080/self.resolution
        accuracy_absolute = [45, 51, 56, 58, 59.5]
        accuracy_percentage = [i/max(accuracy_absolute) for i in accuracy_absolute]
        resolution_absolute = [96, 128, 160, 192, 224]
        resolution_percentage = [i/max(resolution_absolute) for i in resolution_absolute]
        function = np.polyfit(resolution_percentage, accuracy_percentage, 2)
        predicted_accuracy_percentage = np.poly1d(function)
        return predicted_accuracy_percentage(resolution)