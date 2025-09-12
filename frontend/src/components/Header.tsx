import React from 'react';
import { Bell, ChevronDown, TrendingUp } from 'lucide-react';
import { Button } from './ui/button';
import { SimpleThemeToggle } from './theme-toggle';

interface HeaderProps {
  className?: string;
}

export function Header({ className = '' }: HeaderProps) {
  return (
    <header className={`h-16 bg-background border-b border-border flex items-center justify-between px-6 ${className}`}>
      {/* Logo and Page Title */}
      <div className="flex items-center space-x-3">
        {/* Option 1: Using your own logo image */}
        <div className="w-10 h-10 rounded-lg overflow-hidden flex items-center justify-center">
          <img 
            src="/src/components/ui/Kian_Lok_Chin_Create_an_logo_with_light_purple_color_with_the_name_of__TrendAI__b72cc5af-1a1c-4776-87be-3400bcee5099.png" 
            alt="TrendAI Logo" 
            className="w-full h-full object-contain"
          />
        </div>
        
        <h1 className="text-xl font-semibold text-primary">TrendAI</h1>
      </div>

      {/* Actions */}
      <div className="flex items-center space-x-4">
        {/* Theme Toggle */}
        <SimpleThemeToggle />

        {/* Notifications */}
        <Button variant="ghost" size="sm" className="relative p-2 text-muted-foreground hover:text-foreground">
          <Bell className="w-5 h-5" />
          <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
        </Button>

        {/* User Avatar and Dropdown */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
            <span className="text-primary-foreground text-sm font-medium">NB</span>
          </div>
          <div className="hidden md:block">
            <p className="text-sm font-medium text-foreground">WeAreNewbie</p>
            <p className="text-xs text-muted-foreground">Administrator</p>
          </div>
        </div>
      </div>
    </header>
  );
}
