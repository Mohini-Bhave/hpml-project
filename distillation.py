from setfit import DistillationSetFitTrainer

def distill_model(student_model, teacher_model, train_dataset_student):
    
    student_trainer = DistillationSetFitTrainer(
        teacher_model=teacher_model,
        train_dataset=train_dataset_student,
        student_model=student_model,
    )
    student_trainer.train()    
    return student_trainer