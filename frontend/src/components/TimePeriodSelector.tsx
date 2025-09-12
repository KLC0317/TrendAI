import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';

interface TimePeriodSelectorProps {
  selectedPeriod: number;
  onPeriodChange: (period: number) => void;
  className?: string;
}

const periods = [
  { value: 10, label: '10 Days' },
  { value: 20, label: '20 Days' },
  { value: 30, label: '30 Days' },
  { value: 40, label: '40 Days' },
  { value: 50, label: '50 Days' },
  { value: 60, label: '60 Days' },
  { value: 70, label: '70 Days' },
  { value: 80, label: '80 Days' },
  { value: 90, label: '90 Days' },
  { value: 100, label: '100 Days' },
];

export function TimePeriodSelector({ 
  selectedPeriod, 
  onPeriodChange, 
  className = '' 
}: TimePeriodSelectorProps) {
  return (
    <div className={`flex flex-wrap items-center gap-2 bg-muted/50 p-2 rounded-lg ${className}`}>
      {periods.map((period) => (
        <Button
          key={period.value}
          variant={selectedPeriod === period.value ? 'default' : 'ghost'}
          size="sm"
          onClick={() => onPeriodChange(period.value)}
          className={`px-3 py-1.5 text-xs transition-all ${
            selectedPeriod === period.value
              ? 'bg-background text-primary shadow-sm border border-primary/20 hover:bg-background'
              : 'text-muted-foreground hover:text-foreground hover:bg-accent'
          }`}
        >
          {period.label}
        </Button>
      ))}
    </div>
  );
}
