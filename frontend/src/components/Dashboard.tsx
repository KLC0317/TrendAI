import React from 'react';
import { Leaderboard } from './Leaderboard';

interface DashboardProps {
  className?: string;
}

export function Dashboard({ className = '' }: DashboardProps) {
  return (
    <div className={`p-6 ${className}`}>
      <Leaderboard />
    </div>
  );
}