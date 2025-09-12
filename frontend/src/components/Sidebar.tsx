import React from 'react';
import { 
  Home, 
  BarChart3, 
  Users, 
  Settings, 
  FileText, 
  Calendar,
  Bell,
  HelpCircle,
  Brain
} from 'lucide-react';

interface SidebarProps {
  className?: string;
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

const navigationItems = [
  { icon: Home, label: 'Dashboard', key: 'dashboard' },
  { icon: BarChart3, label: 'Analytics', key: 'analytics' },
  { icon: Brain, label: 'AI Insights', key: 'ai-insights' }, // Add this line
  { icon: Users, label: 'Admin', key: 'admins' },
];

export function Sidebar({ className = '', activeTab = 'dashboard', onTabChange }: SidebarProps) {
  return (
    <aside className={`w-64 h-full bg-background border-r border-border flex flex-col ${className}`}>
      {/* Logo Section */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <BarChart3 className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="font-semibold text-foreground">DashBoard</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-1">
          {navigationItems.map((item, index) => {
            const Icon = item.icon;
            const isActive = activeTab === item.key;
            return (
              <li key={index}>
                <button
                  onClick={() => onTabChange?.(item.key)}
                  className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary/10 text-primary border border-primary/20'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
