from setfit import SetFitModel, SetFitTrainer

def train_student(model_name,dataset):
    baseline_model = SetFitModel.from_pretrained(
       model_name
    )
    trainer_baseline_model = SetFitTrainer(
        model=baseline_model, train_dataset=dataset
    )
    trainer_baseline_model.train()
    
    return trainer_baseline_model
    