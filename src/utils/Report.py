from pathlib import Path
import logging
import datetime

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas 
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.pdfbase import pdfmetrics 
from reportlab.lib import colors 
from reportlab.platypus import Table

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


class Report():
    def __init__(self,
                 config: dict,
                 ):
        self.description = config['details']['description']
        self.metrics = config['algorithms']['metrics']
        # self.epochs = str(config['training']['epochs'])
        self.dataset = config['data']['source']['full']
        # self.batch_size = str(config['training']['batch_size'])
        # self.lr = str(config['training']['init_lr'])
        self.dataset_test = config['data']['annotations']['test']
        output_dir = Path(config['run']['output_dir'])
        run_name = config['run']['name']
        output_dir = output_dir / run_name
        report_path = output_dir / 'reports'
        report_path.mkdir(exist_ok=True, parents=True)
        date = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        title = 'Reporte de experimento '
        title += date
        file_name = f'report_{date}.pdf'
        self.pdf = canvas.Canvas(str(report_path / file_name)) 
        self.pdf.setTitle(title) 
        self.pdf.drawCentredString(300, 770, title)

        self.__write_basic_info()         
        

    def __write_basic_info(self):        
        text = self.pdf.beginText(40, 750)        
        self.description = 'Descripcion: ' + self.description 
        stride = 80
        for i in range(0, len(self.description), stride):
            tmp = self.description[i:min(i+stride, len(self.description))]
            text.textLine(tmp)
        self.pdf.drawText(text)
        self.pdf.showPage()
        # logger.info('[Report] description wrote')
        
        text.textLine('Metrics: ')
        if isinstance(self.metrics, dict):
            for key, value in self.metrics.items():
                text.textLine(f"{key}: {value}")
        elif isinstance(self.metrics, list):
            for m in self.metrics:
                text.textLine(f"{m}")
        else:
            text.textLine(f"{self.metrics}")
        # logger.info('[Report] metrics wrote')


    def write(self):        
        self.pdf.save() 


    def write_table(self, data: list) -> None:
        table = Table(data)
        table.wrapOn(self.pdf, 10, 10)
        table.drawOn(self.pdf, 0, 550)
        self.pdf.showPage()

    def write_text(self,
                   text: str,
                   new_page: bool = False) -> None:
        text_pdf = self.pdf.beginText(40, 750)        
        stride = 80
        for i in range(0, len(text), stride):
            tmp = text[i:min(i+stride, len(text))]
            text_pdf.textLine(tmp)
        self.pdf.drawText(text_pdf)
        if new_page:
            self.pdf.showPage()


    def set_plot(self, history, key1, loc_legend='upper right'):
        plt.plot(history.history[key1], label = key1)
        plt.plot(history.history['val_'+key1], label = 'val_'+key1)
        plt.xlabel('Epoch')
        plt.ylabel(key1)
        plt.legend(loc=loc_legend)
        plt.savefig(self.data_path / f'{key1}.png', bbox_inches='tight')
        plt.clf()
        self.plots.append(key1)

    def set_image(self):
        pass

    def generate_model_image(self):
        model = keras.saving.load_model(self.model_trained_dir)
        keras.utils.plot_model(model, 
                               self.data_path / f'model.png', 
                               show_shapes=True,
                               show_layer_activations=True,
                               show_trainable=True)

    def get_best_metrics(self):
        model = keras.saving.load_model(self.model_trained_dir)
        test_ds = loaders.get_data_loader(self.cfg, split='test')
        metrics = model.evaluate(test_ds, verbose=1, return_dict=True)
        return metrics
    
    def get_visual_test(self):
        p = Path(self.dataset_test)
        logger.info(f"Dataset for testing: {str(p)}")

        return make_plot(dataset_dir = p,
                         model_dir = self.model_trained_dir,
                         output_dir = str(self.data_path / f'visual_test.png'),
                         max_size = 63,
                         columns = 7,
                         classification = self.cfg['details']['task'],
                         remove_corner = False,
                         h_target=self.h_target,
                         w_target=self.w_target)