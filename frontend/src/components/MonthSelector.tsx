import React, { useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Calendar } from 'lucide-react';

interface MonthSelectorProps {
  selectedMonth?: string;
  onMonthChange?: (month: string) => void;
  className?: string;
}

export function MonthSelector({ selectedMonth, onMonthChange, className = '' }: MonthSelectorProps) {
  const [months, setMonths] = useState<Array<{ value: string; label: string; date: Date }>>([]);
  const [currentSelectedMonth, setCurrentSelectedMonth] = useState<string>('');

  useEffect(() => {
    // Generate months based on current date
    const generateMonths = () => {
      const now = new Date();
      const currentMonth = now.getMonth(); // 0-based (0 = January, 11 = December)
      const currentYear = now.getFullYear();
      
      const monthNames = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
      ];
      
      const monthOptions = [];
      
      // Add current month and next 2 months
      for (let i = 0; i < 3; i++) {
        const monthIndex = (currentMonth + i) % 12;
        const year = currentYear + Math.floor((currentMonth + i) / 12);
        const date = new Date(year, monthIndex, 1);
        
        monthOptions.push({
          value: `${year}-${String(monthIndex + 1).padStart(2, '0')}`,
          label: `${monthNames[monthIndex]} ${year}`,
          date: date
        });
      }
      
      return monthOptions;
    };

    const monthOptions = generateMonths();
    setMonths(monthOptions);
    
    // Set default selected month to current month
    if (!selectedMonth) {
      const currentMonth = new Date();
      const currentMonthValue = `${currentMonth.getFullYear()}-${String(currentMonth.getMonth() + 1).padStart(2, '0')}`;
      setCurrentSelectedMonth(currentMonthValue);
      onMonthChange?.(currentMonthValue);
    } else {
      setCurrentSelectedMonth(selectedMonth);
    }
  }, [selectedMonth, onMonthChange]);

  const handleMonthChange = (value: string) => {
    setCurrentSelectedMonth(value);
    onMonthChange?.(value);
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <Calendar className="h-4 w-4 text-muted-foreground" />
      <Select value={currentSelectedMonth} onValueChange={handleMonthChange}>
        <SelectTrigger className="w-[200px]">
          <SelectValue placeholder="Select month" />
        </SelectTrigger>
        <SelectContent>
          {months.map((month) => (
            <SelectItem key={month.value} value={month.value}>
              {month.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
