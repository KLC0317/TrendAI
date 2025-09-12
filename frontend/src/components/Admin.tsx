import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Upload, Settings, Calendar, Video, MessageSquare, Tag, Brain, Database } from 'lucide-react';

interface AdminProps {
  className?: string;
}

interface ModelMetadata {
  model_type: string;
  analysis_timestamp: string;
  model_version: string;
  weight_factors: {
    video_weight: number;
    comment_weight: number;
    tag_weight: number;
  };
}

interface RawModelData {
  model_metadata: ModelMetadata;
  creation_date: string;
}

export function Admin({ className = '' }: AdminProps) {
  console.log("Admin component rendered!");
  const [isUploading, setIsUploading] = useState(false);
  
  // Mock data - replace with actual API data
  const modelData: RawModelData = {
    model_metadata: {
      model_type: "EnhancedTrendAI",
      analysis_timestamp: "2025-09-06T14:43:22.310439",
      model_version: "1.0.0",
      weight_factors: {
        video_weight: 0.6,
        comment_weight: 0.25,
        tag_weight: 0.15
      }
    },
    creation_date: "2025-09-06T16:14:18.269639"
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    
    // Simulate upload process
    setTimeout(() => {
      setIsUploading(false);
      alert('Dataset uploaded successfully!');
    }, 2000);
  };

  return (
    <div className={`p-6 space-y-6 bg-background ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Admin Panel</h1>
          <p className="text-muted-foreground mt-1">Model management and configuration</p>
        </div>
      </div>

      {/* Model Information Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-lg font-semibold text-card-foreground">
              <Settings className="w-5 h-5 mr-2 text-blue-500" />
              Model Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center">
                  <Video className="w-4 h-4 mr-2 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">Video Weight Factor</span>
                </div>
                <span className="text-sm font-semibold text-foreground">
                  {(modelData.model_metadata.weight_factors.video_weight * 100).toFixed(0)}%
                </span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center">
                  <MessageSquare className="w-4 h-4 mr-2 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">Comment Weight Factor</span>
                </div>
                <span className="text-sm font-semibold text-foreground">
                  {(modelData.model_metadata.weight_factors.comment_weight * 100).toFixed(0)}%
                </span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div className="flex items-center">
                  <Tag className="w-4 h-4 mr-2 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">Tag Weight Factor</span>
                </div>
                <span className="text-sm font-semibold text-foreground">
                  {(modelData.model_metadata.weight_factors.tag_weight * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            
            <div className="pt-2 border-t border-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="text-sm text-muted-foreground mr-2">Model Type:</span>
                  <span className="text-sm font-medium text-foreground">{modelData.model_metadata.model_type}</span>
                </div>
                <div className="flex items-center">
                  <span className="text-sm text-muted-foreground mr-2">Version:</span>
                  <span className="text-sm font-medium text-foreground">{modelData.model_metadata.model_version}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Last Trained */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center text-lg font-semibold text-card-foreground">
              <Calendar className="w-5 h-5 mr-2 text-green-500" />
              Model Last Trained  
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center mb-2">
                <Brain className="w-5 h-5 mr-2 text-green-500" />
                <span className="text-sm font-medium text-foreground">Last Analysis</span>
              </div>
              <p className="text-lg font-semibold text-foreground">
                {formatDate(modelData.model_metadata.analysis_timestamp)}
              </p>
            </div>
            
            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center mb-2">
                <Database className="w-5 h-5 mr-2 text-blue-500" />
                <span className="text-sm font-medium text-foreground">Model Created</span>
              </div>
              <p className="text-lg font-semibold text-foreground">
                {formatDate(modelData.creation_date)}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Upload New Dataset */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-lg font-semibold text-card-foreground">
            <Upload className="w-5 h-5 mr-2 text-purple-500" />
            Upload New Dataset
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="p-8 border-2 border-dashed border-border rounded-lg text-center hover:border-muted-foreground transition-colors">
              <Upload className="w-8 h-8 mx-auto text-muted-foreground mb-4" />
              <p className="text-sm text-muted-foreground mb-3">
                Drop your dataset files here or click to browse
              </p>
              <p className="text-xs text-muted-foreground mb-6">
                Supported formats: CSV, JSON, TXT
              </p>
              
              <input
                type="file"
                id="dataset-upload"
                className="hidden"
                accept=".csv,.json,.txt"
                onChange={handleFileUpload}
                disabled={isUploading}
              />
              
              <Button
                variant="outline"
                className="rounded-lg px-6 py-2"
                onClick={() => document.getElementById('dataset-upload')?.click()}
                disabled={isUploading}
              >
                {isUploading ? (
                  <>
                    <div className="w-4 h-4 mr-2 border-2 border-muted-foreground border-t-foreground rounded-full animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Choose Files
                  </>
                )}
              </Button>
            </div>
            
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>Last upload: {formatDate(modelData.creation_date)}</span>
              <span>Max file size: 100MB</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
