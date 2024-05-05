from setfit import SetFitModel, SetFitTrainer

def train_teacher(model_name):
    baseline_model = SetFitModel.from_pretrained(
       model_name
    )
    trainer_baseline_model = SetFitTrainer(
        model=baseline_model, train_dataset=train_dataset_teacher
    )
    trainer_baseline_model.train()
    
    return trainer_baseline_model