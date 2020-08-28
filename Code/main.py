import laserbeam
from os.path import exists, join

def main():

    trainer = laserbeam.Trainer()



    if exists(join(trainer.model_path, "checkpoint")):
        trainer.load_checkpoint(join(trainer.model_path, "checkpoint", 'epoch069_2020-08-25_14-22-06.pth'))
        #trainer.load_checkpoint()

    #print(history, current_epoch)
    #train_model(100,20)
    trainer.evaluate()
    #print(len(datasets['test']))
    #example = trainer.datasets['test'].samples_dataframe.iloc[733]
    #output = trainer.test_single_example(example)
    #print(180.0 * output)
    #print(example)
    


if __name__ == '__main__':
    main()

    
