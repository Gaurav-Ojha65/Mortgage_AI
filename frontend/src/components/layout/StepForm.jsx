import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import Input from '../ui/Input';
import Button from '../ui/Button';
import './StepForm.css';

const formSchema = z.object({
  income: z.number().min(1000, "Income must be at least 1,000"),
  loan_amount: z.number().min(10000, "Loan amount must be at least 10,000"),
  interest_rate: z.number().min(0.1, "Must be > 0").max(30, "Cannot exceed 30%"),
  loan_term: z.number().min(1, "Minimum 1 year").max(40, "Maximum 40 years"),
  credit_score: z.number().min(300, "Minimum 300").max(850, "Maximum 850"),
  existing_loans: z.number().min(0, "Cannot be negative"),
  age_band: z.string().min(1, "Required"),
  region: z.string().min(1, "Required"),
});


export default function StepForm({ onSubmit, isPredicting }) {
  const [step, setStep] = useState(1);
  
  const { register, handleSubmit, trigger, formState: { errors } } = useForm({
    resolver: zodResolver(formSchema),
    mode: 'onTouched',
    defaultValues: {
      income: 5000,
      loan_amount: 250000,
      interest_rate: 6.5,
      loan_term: 30,
      credit_score: 720,
      existing_loans: 0,
      age_band: '26-35',
      region: 'North'
    }

  });

  const nextStep = async () => {
    let fieldsToValidate = [];
    if (step === 1) fieldsToValidate = ['income', 'existing_loans'];
    if (step === 2) fieldsToValidate = ['loan_amount', 'loan_term', 'interest_rate'];
    if (step === 3) fieldsToValidate = ['credit_score', 'age_band', 'region'];


    const isValid = await trigger(fieldsToValidate);
    if (isValid) setStep(s => Math.min(s + 1, 3));
  };

  const prevStep = () => setStep(s => Math.max(s - 1, 1));

  const handleFormSubmit = (data) => {
    onSubmit(data);
  };

  return (
    <div className="step-form-container">
      <div className="form-progress">
        <div className={`progress-step ${step >= 1 ? 'active' : ''}`}>1. Income</div>
        <div className={`progress-line ${step >= 2 ? 'active' : ''}`} />
        <div className={`progress-step ${step >= 2 ? 'active' : ''}`}>2. Loan Details</div>
        <div className={`progress-line ${step >= 3 ? 'active' : ''}`} />
        <div className={`progress-step ${step >= 3 ? 'active' : ''}`}>3. Final Info</div>

      </div>

      <form onSubmit={handleSubmit(handleFormSubmit)} className="form-content">
        {step === 1 && (
          <div className="form-step slide-in">
            <h3 className="step-title">Financial Details</h3>
            <Input
              label="Monthly Income ($)"
              type="number"
              {...register("income", { valueAsNumber: true })}
              error={errors.income?.message}
              placeholder="e.g. 5000"
            />
            <Input
              label="Existing Loans Count"
              type="number"
              {...register("existing_loans", { valueAsNumber: true })}
              error={errors.existing_loans?.message}
            />
          </div>
        )}

        {step === 2 && (
          <div className="form-step slide-in">
            <h3 className="step-title">Loan Details</h3>
            <Input
              label="Loan Amount ($)"
              type="number"
              {...register("loan_amount", { valueAsNumber: true })}
              error={errors.loan_amount?.message}
            />
            <div className="form-row">
              <Input
                label="Interest Rate (%)"
                type="number"
                step="0.1"
                {...register("interest_rate", { valueAsNumber: true })}
                error={errors.interest_rate?.message}
              />
              <Input
                label="Term (Years)"
                type="number"
                {...register("loan_term", { valueAsNumber: true })}
                error={errors.loan_term?.message}
              />
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="form-step slide-in">
            <h3 className="step-title">Final Details</h3>
            <Input
              label="Credit Score"
              type="number"
              {...register("credit_score", { valueAsNumber: true })}
              error={errors.credit_score?.message}
              helperText="FICO Score between 300 and 850"
            />
            <div className="form-row">
              <div className="ui-input-wrapper">
                <label className="ui-label">Age Band</label>
                <select className="ui-input" {...register("age_band")}>
                  <option value="18-25">18-25</option>
                  <option value="26-35">26-35</option>
                  <option value="36-45">36-45</option>
                  <option value="46-60">46-60</option>
                  <option value="60+">60+</option>
                </select>
                {errors.age_band && <span className="ui-error-text">{errors.age_band.message}</span>}
              </div>
              <div className="ui-input-wrapper">
                <label className="ui-label">Region</label>
                <select className="ui-input" {...register("region")}>
                  <option value="North">North</option>
                  <option value="South">South</option>
                  <option value="East">East</option>
                  <option value="West">West</option>
                  <option value="Central">Central</option>
                </select>
                {errors.region && <span className="ui-error-text">{errors.region.message}</span>}
              </div>
            </div>
          </div>
        )}


        <div className="form-actions">
          {step > 1 && (
            <Button type="button" variant="secondary" onClick={prevStep} disabled={isPredicting}>
              Back
            </Button>
          )}
          {step < 3 ? (
            <Button type="button" onClick={nextStep} style={{ marginLeft: 'auto' }}>
              Next Step
            </Button>
          ) : (
            <Button type="submit" loading={isPredicting} style={{ marginLeft: 'auto' }}>
              Analyze Risk
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
