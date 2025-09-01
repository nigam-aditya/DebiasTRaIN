'''
PARTS I ADDED START
'''

@app.command()
def initial_eval(model_name: str, prompt_length: int, experiment_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    evaluator = BiasEvaluatorForBert()

    results = evaluator.evaluate(
        model, tokenizer,
        prompt_length=prompt_length,
        return_embeddings=False,
        return_words_close_to_prompts=True,
        return_stereo_set_results=True,
        position_id_adjustment=PositionIdAdjustmentType.none,
    )

    out_dir = f"./runs/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "initial_eval.json"), "w") as fout:
        json.dump(results, fout, indent=2)
    
    dump_predictions_on_training_dataset(model, tokenizer, 10, prompt_length, out_file_sents=f"./runs/{experiment_name}/predictions_before.html")
    console.log(f"Saved initial evaluation to {out_dir}/initial_eval.json")

@app.command("train")
def train_only(
    model_name='bert-base-uncased',
    prompt_length: int = 10,
    experiment_name: str = "test",
    num_epochs: int = 20,
    loss_type: LossFunctionType = LossFunctionType.equal_valid_options_mask_logits,
    position_ids_adjustment: PositionIdAdjustmentType = PositionIdAdjustmentType.none,
    gender_specific_options: bool = False,
    use_names: bool = False,
    prompt_init_text: str = None,
):
    is_roberta = "roberta" in model_name
    writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")
    console.log(f"Writing tensorboard logs to {writer.log_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    occupation_dataset = prepare_dataset_for_masked_model(tokenizer, False, model, use_names=use_names, prepare_for_roberta=is_roberta)
    occupation_dataset.set_format("torch")

    if prompt_init_text:
        peft_config = peft.PromptTuningConfig(
            task_type="SEQ_CLS", num_virtual_tokens=prompt_length,
            prompt_tuning_init=peft.PromptTuningInit.TEXT,
            prompt_tuning_init_text=prompt_init_text,
            tokenizer_name_or_path=model_name
        )
    else:
        peft_config = peft.PromptTuningConfig(
            task_type="SEQ_CLS", num_virtual_tokens=prompt_length)

    model = peft.get_peft_model(model, peft_config).to(device)
    model.train()

    train_loader = DataLoader(occupation_dataset, batch_size=16, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.06 * (len(train_loader) * num_epochs),
        num_training_steps=(len(train_loader) * num_epochs),
    )

    def compute_loss(batch):
        input_ids_male = batch["input_ids_male"].to(device)
        input_ids_female = batch["input_ids_female"].to(device)
        attention_mask_male = batch["attention_mask_male"].to(device)
        attention_mask_female = batch["attention_mask_female"].to(device)
        output_indices = batch["output_indices"].to(device)
        mask_token_index_male = batch["mask_token_idx_male"].to(device)
        mask_token_index_female = batch["mask_token_idx_female"].to(device)
        male_original = batch["male_original_model_mask_logits"].to(device)
        female_original = batch["female_original_model_mask_logits"].to(device)

        pos_ids_male = create_positional_ids(input_ids_male.size(0), input_ids_male.size(1), prompt_length, device, position_ids_adjustment)
        pos_ids_female = create_positional_ids(input_ids_female.size(0), input_ids_female.size(1), prompt_length, device, position_ids_adjustment)

        output_male = model(input_ids_male, attention_mask=attention_mask_male, position_ids=pos_ids_male)
        output_female = model(input_ids_female, attention_mask=attention_mask_female, position_ids=pos_ids_female)

        mask_token_index_male += prompt_length
        mask_token_index_female += prompt_length

        male_logits = output_male.logits[torch.arange(output_male.logits.size(0)), mask_token_index_male]
        female_logits = output_female.logits[torch.arange(output_female.logits.size(0)), mask_token_index_female]

        if loss_type == LossFunctionType.equal_valid_options_mask_logits:
            return loss_equal_valid_options_mask_logits(male_logits, female_logits, output_indices)
        elif loss_type == LossFunctionType.mean_valid_options_original_probabilities:
            return loss_mean_valid_options_original_probabilities(
                male_logits, female_logits, male_original, female_original,
                output_indices)
        raise RuntimeError("Loss type not supported")

    for epoch in range(1, num_epochs + 1):
        losses = []
        for batch in train_loader:
            optim.zero_grad()
            loss = compute_loss(batch)
            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.item())
        writer.add_scalar("Loss/train", np.mean(losses), epoch)
        console.log(f"Epoch {epoch}: loss={np.mean(losses):.4f}")

    writer.flush()
    writer.close()
    model.save_pretrained(f"./runs/{experiment_name}/model")
    tokenizer.save_pretrained(f"./runs/{experiment_name}/model")
    console.log(f"Saved model to ./runs/{experiment_name}/model")

@app.command()
def final_eval(experiment_name: str, prompt_length: int):
    model_dir = f"./runs/{experiment_name}/model"
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    evaluator = BiasEvaluatorForBert()
    results = evaluator.evaluate(
        model, tokenizer,
        prompt_length=prompt_length,
        return_embeddings=False,
        return_words_close_to_prompts=True,
        return_stereo_set_results=True,
        position_id_adjustment=PositionIdAdjustmentType.none,
    )
    with open(f"./runs/{experiment_name}/final_eval.json", "w") as fout:
        json.dump(results, fout, indent=2)
    
    dump_predictions_on_training_dataset(model, tokenizer, 10, prompt_length, out_file_sents=f"./runs/{experiment_name}/predictions_after.html")
    console.log(f"Saved final evaluation to ./runs/{experiment_name}/final_eval.json")

@app.command()
def compare(experiment_name: str):
    with open(f"./runs/{experiment_name}/initial_eval.json") as f1:
        initial = json.load(f1)
    with open(f"./runs/{experiment_name}/final_eval.json") as f2:
        final = json.load(f2)

    summary = compute_summary_results(initial, final)
    with open(f"./runs/{experiment_name}/comparison.json", "w") as fout:
        json.dump(summary, fout, indent=2)
    console.print(summary)


'''
PARTS I ADDED END
'''