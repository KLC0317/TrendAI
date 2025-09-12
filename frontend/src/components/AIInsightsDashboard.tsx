import MDEditor from '@uiw/react-md-editor';
import '@uiw/react-md-editor/markdown-editor.css';
import '@uiw/react-markdown-preview/markdown.css';
import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';
import { 
  FileText, 
  Brain, 
  TrendingUp, 
  BarChart3, 
  Download, 
  Search,
  Lightbulb,
  AlertTriangle,
  CheckCircle,
  Info,
  Zap,
  Target,
  Database,
  Loader2,
  Copy,
  Share2,
  Settings,
  RefreshCw,
  Eye,
  Filter,
  ChevronDown,
  ChevronUp,
  ToggleLeft,
  ToggleRight,
  Globe
} from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';

interface JsonFile {
  id: string;
  name: string;
  path: string;
  data: any;
  lastAnalyzed?: Date;
}

interface Insight {
  id: string;
  type: 'trend' | 'pattern' | 'anomaly' | 'opportunity' | 'risk' | 'summary';
  title: string;
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high';
  category: string;
  details?: string;
  recommendations?: string[];
  keyFindings?: string[];
  dataSource: string;
  webVerified?: boolean;
}

interface AnalysisConfig {
  focusAreas: string[];
  analysisDepth: 'quick' | 'standard' | 'deep';
  includeRecommendations: boolean;
  customPrompt?: string;
  useWebVerification: boolean;
}

// Storage key constants
const STORAGE_KEYS = {
  JSON_FILES: 'loreal-ai-json-files',
  SELECTED_FILE: 'loreal-ai-selected-file',
  INSIGHTS: 'loreal-ai-insights',
  ANALYSIS_CONFIG: 'loreal-ai-analysis-config',
  EXPANDED_INSIGHT: 'loreal-ai-expanded-insight',
  SEARCH_QUERY: 'loreal-ai-search-query',
  FILTER_TYPE: 'loreal-ai-filter-type',
  SHOW_CONFIG: 'loreal-ai-show-config'
};

// Storage utility functions
const saveToStorage = (key: string, data: any) => {
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch (error) {
    console.warn('Failed to save to localStorage:', error);
  }
};

const loadFromStorage = (key: string, defaultValue: any = null) => {
  try {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : defaultValue;
  } catch (error) {
    console.warn('Failed to load from localStorage:', error);
    return defaultValue;
  }
};

// Hardcoded API key - replace with your actual key
const GEMINI_API_KEY = 'AIzaSyBtmaJDhf5rEqOnGukqKRvo1_XeDtS34Ic';

// Single JSON file - only forecast data
const AVAILABLE_JSON_FILES = [
  { name: 'Forecast Data', path: '/forecast_data_only_20250906_161418.json' },
];

const defaultConfig: AnalysisConfig = {
  focusAreas: [],
  analysisDepth: 'standard',
  includeRecommendations: true,
  useWebVerification: false
};

// Enhanced L'Oréal Beauty Trend Analysis Prompt
const LOREAL_BEAUTY_PROMPT = `
# L'Oréal Beauty Trend Intelligence Analysis

You are an expert beauty industry analyst specializing in trend identification and strategic forecasting for L'Oréal. Analyze the provided YouTube beauty trend forecast data to deliver actionable insights that align with L'Oréal's business objectives.

## Analysis Framework

### 1. TREND IDENTIFICATION & CLASSIFICATION
- **Emerging vs. Established**: Classify each trend by growth trajectory and current market position
- **Beauty Relevance Scoring**: Rate each trend's direct applicability to beauty/cosmetics (1-10 scale)
- **Cross-Category Potential**: Identify trends that could bridge beauty with lifestyle, wellness, or cultural movements
- **Geographic Insights**: Analyze regional/cultural trend patterns (note keywords like "india," "russian," "pakistan")

### 2. GENERATIONAL SEGMENTATION ANALYSIS
For each significant trend, identify:
- **Primary Generation**: Which generation (Gen Z, Millennial, Gen X, Boomer) is driving this trend
- **Adoption Patterns**: How the trend spreads across generational cohorts
- **Language Indicators**: Generational markers in keywords and engagement patterns
- **Platform Preferences**: Infer which platforms each generation uses for trend discovery

### 3. TREND LIFECYCLE & TIMING ANALYSIS
- **Growth Phase**: Analyze growth rate patterns to determine trend maturity
- **Peak Prediction**: Forecast when trends will reach maximum engagement
- **Decay Indicators**: Identify early warning signs of trend decline
- **Optimal Entry Window**: Recommend the best timing for L'Oréal to engage with each trend

### 4. STRATEGIC RECOMMENDATIONS

#### Immediate Action Items (Next 30 Days):
- High-growth, beauty-relevant trends requiring immediate attention
- Content creation opportunities
- Influencer partnership recommendations

#### Medium-term Strategy (30-90 Days):
- Product development insights
- Marketing campaign themes
- Regional market prioritization

#### Long-term Vision (3+ Months):
- Innovation pipeline recommendations
- Brand positioning adjustments
- Emerging market opportunities

### 5. RISK ASSESSMENT
- **Cultural Sensitivity**: Flag trends that may require careful cultural navigation
- **Brand Alignment**: Assess compatibility with L'Oréal's brand values and positioning
- **Competition Analysis**: Evaluate competitive landscape for each trend
- **Investment vs. ROI**: Recommend resource allocation priorities

## Key Questions to Address:

1. **Which trends represent the highest opportunity for L'Oréal specifically?**
2. **What are the key demographic segments participating in each trend?**
3. **When is the optimal time to enter each trend to maximize impact?**
4. **Which trends are too saturated or declining to pursue?**
5. **What cultural or regional adaptations are needed for global trends?**
6. **How do these trends align with current L'Oréal product categories?**
7. **What new product opportunities do these trends suggest?**

## Web Search Enhancement Instructions:
If web verification is enabled, please search for current information about the identified trends to:
- Verify trend authenticity and current status on platforms like TikTok, Instagram, YouTube
- Find recent examples of beauty brands engaging with these trends
- Identify influencers and content creators driving these trends
- Discover related emerging trends not captured in the forecast data
- Validate cultural and regional trend insights
- Check for any recent developments that might affect trend trajectory
- Assess competition and market saturation for each trend

For each major trend identified, search for:
- "beauty trends 2024 2025 [trend keywords]"
- "[trend keywords] makeup skincare beauty influencer"
- "Gen Z millennial beauty trends [trend keywords]"
- "TikTok Instagram beauty trend [trend keywords]"
- "L'Oréal Maybelline competitors [trend keywords]"
- "[trend keywords] beauty market analysis"

## Output Format:

### Executive Summary
- Top 3 Priority Trends for L'Oréal
- Key Risk Factors
- Overall Market Momentum Assessment

### Detailed Trend Analysis
For each major trend:
- **Trend Name & Description**
- **Beauty Industry Relevance Score (1-10)**
- **Web Verification Status** (Confirmed/Emerging/Declining/Unverified)
- **Primary Target Demographic**
- **Growth Trajectory Analysis**
- **Optimal Engagement Timeline**
- **Strategic Recommendations**
- **Risk Factors**
- **Competition Landscape** (if web verified)

### Strategic Action Plan
- **Immediate Priorities** (30 days)
- **Medium-term Initiatives** (90 days)
- **Long-term Innovation Opportunities** (6+ months)

### Cultural & Regional Insights
- Geographic trend variations
- Cultural adaptation requirements
- Regional market prioritization

### Web Intelligence Summary (if enabled)
- Trend verification results
- Additional trends discovered
- Competitive intelligence gathered
- Influencer landscape insights

Format your response with clear headers using ## for main sections and * for bullet points. Be specific and provide actionable insights that transform this raw trend data into business intelligence for L'Oréal's beauty division.
`;

export function AIInsightsDashboard({ className = '' }: { className?: string }) {
  // Initialize state with localStorage values
  const [jsonFiles, setJsonFiles] = useState<JsonFile[]>(() => 
    loadFromStorage(STORAGE_KEYS.JSON_FILES, [])
  );
  const [selectedFile, setSelectedFile] = useState<string | null>(() => 
    loadFromStorage(STORAGE_KEYS.SELECTED_FILE, null)
  );
  const [insights, setInsights] = useState<Insight[]>(() => 
    loadFromStorage(STORAGE_KEYS.INSIGHTS, [])
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>(() => 
    loadFromStorage(STORAGE_KEYS.ANALYSIS_CONFIG, defaultConfig)
  );
  const [searchQuery, setSearchQuery] = useState(() => 
    loadFromStorage(STORAGE_KEYS.SEARCH_QUERY, '')
  );
  const [filterType, setFilterType] = useState<string>(() => 
    loadFromStorage(STORAGE_KEYS.FILTER_TYPE, 'all')
  );
  const [selectedInsight, setSelectedInsight] = useState<Insight | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedInsight, setExpandedInsight] = useState<string | null>(() => 
    loadFromStorage(STORAGE_KEYS.EXPANDED_INSIGHT, null)
  );
  const [showConfig, setShowConfig] = useState(() => 
    loadFromStorage(STORAGE_KEYS.SHOW_CONFIG, false)
  );

  // Persist state changes to localStorage
  useEffect(() => {
    saveToStorage(STORAGE_KEYS.JSON_FILES, jsonFiles);
  }, [jsonFiles]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.SELECTED_FILE, selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.INSIGHTS, insights);
  }, [insights]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.ANALYSIS_CONFIG, analysisConfig);
  }, [analysisConfig]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.EXPANDED_INSIGHT, expandedInsight);
  }, [expandedInsight]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.SEARCH_QUERY, searchQuery);
  }, [searchQuery]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.FILTER_TYPE, filterType);
  }, [filterType]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.SHOW_CONFIG, showConfig);
  }, [showConfig]);

  // Enhanced markdown to JSX converter with better visual formatting
const parseMarkdownToJSX = (text: string): JSX.Element[] => {
  if (!text) return [];
  
  const lines = text.split('\n');
  const elements: JSX.Element[] = [];
  let currentListItems: string[] = [];
  let currentTable: string[][] = [];
  let listKey = 0;
  let tableKey = 0;
  let codeBlockContent = '';
  let isInCodeBlock = false;
  let codeBlockKey = 0;

  const flushList = () => {
    if (currentListItems.length > 0) {
      elements.push(
        <div key={`list-container-${listKey++}`} className="my-4">
          <ul className="space-y-2">
            {currentListItems.map((item, idx) => (
              <li key={idx} className="flex items-start space-x-3 p-2 rounded-lg bg-gradient-to-r from-blue-50/30 to-purple-50/30 dark:from-blue-950/20 dark:to-purple-950/20 border-l-3 border-blue-400">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                <span className="text-sm leading-relaxed text-foreground">
                  {parseInlineFormatting(item.replace(/^\*\s*/, ''))}
                </span>
              </li>
            ))}
          </ul>
        </div>
      );
      currentListItems = [];
    }
  };

  const flushTable = () => {
    if (currentTable.length > 0) {
      elements.push(
        <div key={`table-container-${tableKey++}`} className="my-6 overflow-x-auto">
          <div className="inline-block min-w-full shadow-lg rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
            <table className="min-w-full">
              <thead className="bg-gradient-to-r from-purple-600 to-blue-600 text-white">
                <tr>
                  {currentTable[0].map((header, idx) => (
                    <th key={idx} className="px-4 py-3 text-left text-sm font-semibold tracking-wider">
                      {parseInlineFormatting(header)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {currentTable.slice(1).map((row, rowIdx) => (
                  <tr key={rowIdx} className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                    {row.map((cell, cellIdx) => (
                      <td key={cellIdx} className="px-4 py-3 text-sm text-gray-900 dark:text-gray-300">
                        {parseInlineFormatting(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
      currentTable = [];
    }
  };

  const parseInlineFormatting = (text: string): React.ReactNode => {
    // Handle bold text
    const boldRegex = /\*\*(.*?)\*\*/g;
    let parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;

    while ((match = boldRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      parts.push(
        <strong key={`bold-${match.index}`} className="font-bold text-purple-700 dark:text-purple-300">
          {match[1]}
        </strong>
      );
      lastIndex = match.index + match[0].length;
    }
    
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    // Handle inline code
    return parts.map((part, idx) => {
      if (typeof part === 'string') {
        const codeRegex = /`([^`]+)`/g;
        const codeParts: React.ReactNode[] = [];
        let lastCodeIndex = 0;
        let codeMatch;

        while ((codeMatch = codeRegex.exec(part)) !== null) {
          if (codeMatch.index > lastCodeIndex) {
            codeParts.push(part.substring(lastCodeIndex, codeMatch.index));
          }
          codeParts.push(
            <code key={`code-${idx}-${codeMatch.index}`} className="px-2 py-1 bg-gray-100 dark:bg-gray-800 text-purple-600 dark:text-purple-400 rounded text-xs font-mono">
              {codeMatch[1]}
            </code>
          );
          lastCodeIndex = codeMatch.index + codeMatch[0].length;
        }

        if (lastCodeIndex < part.length) {
          codeParts.push(part.substring(lastCodeIndex));
        }

        return codeParts.length > 1 ? codeParts : part;
      }
      return part;
    });
  };

  lines.forEach((line, index) => {
    const trimmedLine = line.trim();
    
    // Handle code blocks
    if (trimmedLine.startsWith('```')) {
      if (isInCodeBlock) {
        // End code block
        elements.push(
          <div key={`codeblock-${codeBlockKey++}`} className="my-4">
            <div className="bg-gray-900 dark:bg-gray-800 rounded-lg overflow-hidden border border-gray-600">
              <div className="bg-gradient-to-r from-gray-700 to-gray-800 px-4 py-2 text-xs text-gray-300 font-semibold">
                Code Block
              </div>
              <pre className="p-4 text-sm text-green-400 overflow-x-auto">
                <code>{codeBlockContent}</code>
              </pre>
            </div>
          </div>
        );
        codeBlockContent = '';
        isInCodeBlock = false;
      } else {
        // Start code block
        flushList();
        flushTable();
        isInCodeBlock = true;
      }
      return;
    }

    if (isInCodeBlock) {
      codeBlockContent += line + '\n';
      return;
    }

    if (!trimmedLine) {
      flushList();
      flushTable();
      return;
    }

    // Handle tables
    if (trimmedLine.includes('|') && trimmedLine.split('|').length > 2) {
      const cells = trimmedLine.split('|').map(cell => cell.trim()).filter(cell => cell);
      if (cells.length > 0) {
        currentTable.push(cells);
        return;
      }
    } else if (currentTable.length > 0) {
      flushTable();
    }

    // Handle headers
    if (trimmedLine.startsWith('####')) {
      flushList();
      elements.push(
        <div key={`h4-container-${index}`} className="my-4">
          <h4 className="text-md font-bold text-foreground bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 p-3 rounded-lg border-l-4 border-blue-500">
            {parseInlineFormatting(trimmedLine.replace(/^####\s*/, ''))}
          </h4>
        </div>
      );
    } else if (trimmedLine.startsWith('###')) {
      flushList();
      elements.push(
        <div key={`h3-container-${index}`} className="my-5">
          <h3 className="text-lg font-bold text-foreground bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 p-4 rounded-lg border-l-4 border-purple-500 shadow-sm">
            {parseInlineFormatting(trimmedLine.replace(/^###\s*/, ''))}
          </h3>
        </div>
      );
    } else if (trimmedLine.startsWith('##')) {
      flushList();
      elements.push(
        <div key={`h2-container-${index}`} className="my-6">
          <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-4 rounded-t-lg">
            <h2 className="text-xl font-bold flex items-center">
              <div className="w-3 h-3 bg-white rounded-full mr-3"></div>
              {trimmedLine.replace(/^##\s*/, '')}
            </h2>
          </div>
          <div className="h-1 bg-gradient-to-r from-purple-600 to-blue-600"></div>
        </div>
      );
    } else if (trimmedLine.startsWith('#')) {
      flushList();
      elements.push(
        <div key={`h1-container-${index}`} className="my-6">
          <div className="bg-gradient-to-r from-purple-700 to-blue-700 text-white p-6 rounded-lg shadow-lg">
            <h1 className="text-2xl font-bold text-center">
              {trimmedLine.replace(/^#\s*/, '')}
            </h1>
          </div>
        </div>
      );
    } else if (trimmedLine.match(/^[\*\-\+]\s+/)) {
      currentListItems.push(trimmedLine);
    } else if (trimmedLine.match(/^\d+\.\s+/)) {
      flushList();
      const match = trimmedLine.match(/^(\d+)\.\s+(.+)$/);
      if (match) {
        elements.push(
          <div key={`numbered-${index}`} className="flex items-start space-x-3 my-3 p-3 bg-gradient-to-r from-green-50/40 to-blue-50/40 dark:from-green-950/20 dark:to-blue-950/20 rounded-lg border-l-3 border-green-500">
            <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
              {match[1]}
            </div>
            <span className="text-sm leading-relaxed text-foreground">
              {parseInlineFormatting(match[2])}
            </span>
          </div>
        );
      }
    } else if (trimmedLine.startsWith('>')) {
      flushList();
      elements.push(
        <div key={`quote-${index}`} className="my-4">
          <blockquote className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-950/30 dark:to-orange-950/30 border-l-4 border-yellow-500 p-4 rounded-r-lg shadow-sm">
            <p className="text-sm leading-relaxed text-yellow-800 dark:text-yellow-200 italic">
              {parseInlineFormatting(trimmedLine.replace(/^>\s*/, ''))}
            </p>
          </blockquote>
        </div>
      );
    } else if (trimmedLine.match(/^-{3,}|^={3,}/)) {
      // Horizontal rule
      elements.push(
        <div key={`hr-${index}`} className="my-6">
          <hr className="border-0 h-px bg-gradient-to-r from-transparent via-gray-400 to-transparent" />
        </div>
      );
    } else {
      flushList();
      // Regular paragraph with enhanced styling
      elements.push(
        <div key={`p-container-${index}`} className="my-3">
          <p className="text-sm leading-relaxed text-foreground bg-white/50 dark:bg-gray-900/50 p-3 rounded-lg border border-gray-100 dark:border-gray-800 shadow-sm">
            {parseInlineFormatting(trimmedLine)}
          </p>
        </div>
      );
    }
  });

  flushList();
  flushTable();
  
  return elements;
};

const MarkdownRenderer = ({ content }: { content: string }) => (
  <div data-color-mode="auto">
    <MDEditor.Markdown 
      source={content}
      style={{ 
        backgroundColor: 'transparent',
        color: 'inherit'
      }}
    />
  </div>
);

  // Load JSON files from public folder
  const loadJsonFiles = useCallback(async () => {
    setIsLoadingFiles(true);
    const loadedFiles: JsonFile[] = [];

    for (const file of AVAILABLE_JSON_FILES) {
      try {
        console.log(`Attempting to load: ${file.path}`);
        const response = await fetch(file.path);
        if (response.ok) {
          const data = await response.json();
          const newFile = {
            id: `file-${Date.now()}-${Math.random()}`,
            name: file.name,
            path: file.path,
            data
          };
          loadedFiles.push(newFile);
          console.log(`Successfully loaded: ${file.name}`);
        } else {
          console.warn(`Failed to load ${file.name}: ${response.status} ${response.statusText}`);
        }
      } catch (error) {
        console.warn(`Failed to load ${file.name}:`, error);
      }
    }

    console.log('Total files loaded:', loadedFiles.length);
    setJsonFiles(loadedFiles);
    
    if (loadedFiles.length > 0 && !selectedFile) {
      setSelectedFile(loadedFiles[0].id);
      console.log('Auto-selected file:', loadedFiles[0].id);
    }
    setIsLoadingFiles(false);
  }, [selectedFile]);

  useEffect(() => {
    // Only load files if we don't have any in storage or if storage is empty
    if (jsonFiles.length === 0) {
      loadJsonFiles();
    }
  }, [loadJsonFiles, jsonFiles.length]);

  // Google Gemini API integration with web search
  const callGeminiAPI = async (prompt: string, jsonData: any, useWebSearch: boolean = false): Promise<string> => {
    if (!GEMINI_API_KEY) {
      throw new Error('Gemini API key not configured. Please add your API key to the code.');
    }

    const dataString = JSON.stringify(jsonData, null, 2);
    const truncatedData = dataString.length > 15000 ? 
      dataString.substring(0, 15000) + '...\n[Data truncated for analysis]' : 
      dataString;

    const enhancedPrompt = useWebSearch ? 
      `${prompt}\n\n**WEB SEARCH ENABLED: Please search the web to verify trends, find additional relevant information, check current status of identified trends on social media platforms, and gather competitive intelligence. Include web search findings and verification status in your analysis.**\n\nJSON Data to analyze:\n${truncatedData}` 
      : `${prompt}\n\nJSON Data to analyze:\n${truncatedData}`;

    const requestBody = {
      contents: [{
        parts: [{
          text: enhancedPrompt
        }]
      }],
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
      }
    };

    // Add search grounding if web search is enabled
    if (useWebSearch) {
        requestBody.tools = [
      { googleSearch: {} }
    ];
    }

    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response generated';
  };

  // L'Oréal Beauty Analysis with web verification
  const analyzeWithLorealPrompt = async (fileData: any, config: AnalysisConfig, fileName: string): Promise<Insight[]> => {
    const insights: Insight[] = [];
    
    try {
      // Primary L'Oréal Beauty Analysis
      const lorealResponse = await callGeminiAPI(LOREAL_BEAUTY_PROMPT, fileData, config.useWebVerification);
      
      // Determine confidence based on web verification
      const baseConfidence = 90;
      const webVerifiedConfidence = config.useWebVerification ? 95 : baseConfidence;
      
      insights.push({
        id: `loreal-primary-${Date.now()}`,
        type: 'opportunity',
        title: 'L\'Oréal Beauty Trend Intelligence Report',
        description: 'Comprehensive beauty industry analysis with trend identification, generational insights, and strategic recommendations',
        confidence: webVerifiedConfidence,
        impact: 'high',
        category: 'Strategic Intelligence',
        details: lorealResponse,
        dataSource: fileName,
        webVerified: config.useWebVerification
      });

      // Competitive Analysis (if web verification enabled)
      if (config.useWebVerification) {
        const competitivePrompt = `
Based on the trend data provided and your web search capabilities, conduct a competitive analysis specifically for L'Oréal:

## Competitive Intelligence Framework:
1. **Direct Competitors**: Analyze how Maybelline, Urban Decay, Lancôme, and other L'Oréal brands are engaging with identified trends
2. **Indirect Competitors**: Assess Fenty Beauty, Rare Beauty, Glossier, and emerging beauty brands
3. **Trend Adoption Speed**: Who's first to market with new trends?
4. **Content Strategy**: How competitors are leveraging identified trends in their marketing
5. **Influencer Partnerships**: Key beauty influencers working with competitors on these trends

## Search Focus:
- Recent campaign launches by competitors
- Social media engagement around identified trends
- Product launches aligned with forecasted trends
- Influencer collaboration announcements

Provide specific examples and actionable competitive intelligence.
        `;

        const competitiveResponse = await callGeminiAPI(competitivePrompt, fileData, true);
        
        insights.push({
          id: `competitive-${Date.now()}`,
          type: 'risk',
          title: 'Competitive Intelligence & Market Positioning',
          description: 'Analysis of competitor activities and market positioning opportunities',
          confidence: 92,
          impact: 'high',
          category: 'Competitive Analysis',
          details: competitiveResponse,
          dataSource: fileName,
          webVerified: true
        });
      }

      // Trend Verification Analysis (if web verification enabled)
      if (config.useWebVerification) {
        const verificationPrompt = `
Using web search, verify the authenticity and current status of the top trending topics identified in the forecast data:

## Verification Framework:
1. **Platform Validation**: Check if trends are actually trending on TikTok, Instagram, YouTube
2. **Influencer Adoption**: Identify key beauty influencers discussing these trends
3. **Brand Engagement**: Find examples of beauty brands already engaging with these trends
4. **Search Volume**: Assess current search interest and momentum
5. **Regional Variation**: Confirm geographic trend patterns

## Truth Assessment:
- **Confirmed Trends**: Currently active and growing
- **Emerging Trends**: Early stage but gaining momentum  
- **Declining Trends**: Past peak or losing interest
- **False Signals**: Not actually trending despite data indication

Provide verification status for each major trend and explain discrepancies.
        `;

        const verificationResponse = await callGeminiAPI(verificationPrompt, fileData, true);
        
        insights.push({
          id: `verification-${Date.now()}`,
          type: 'summary',
          title: 'Trend Verification & Authenticity Assessment',
          description: 'Web-based verification of forecasted trends with current market reality',
          confidence: 94,
          impact: 'high',
          category: 'Trend Verification',
          details: verificationResponse,
          dataSource: fileName,
          webVerified: true
        });
      }

      // Custom Focus Areas (if specified)
      if (config.focusAreas.length > 0) {
        const focusPrompt = `
Analyze the L'Oréal beauty trend data with specific focus on: ${config.focusAreas.join(', ')}.

For each focus area, provide:
1. **Current Trend Status**: How these areas are represented in the forecast data
2. **Beauty Industry Application**: Specific opportunities for L'Oréal
3. **Competitive Landscape**: Who's winning in these focus areas
4. **Strategic Recommendations**: Actionable next steps
5. **Success Metrics**: How to measure progress

${config.useWebVerification ? 'Use web search to validate findings and discover additional insights in these focus areas.' : ''}
        `;

        const focusResponse = await callGeminiAPI(focusPrompt, fileData, config.useWebVerification);
        
        insights.push({
          id: `focus-${Date.now()}`,
          type: 'pattern',
          title: `Focus Area Analysis: ${config.focusAreas.join(', ')}`,
          description: 'Deep dive analysis of specified focus areas',
          confidence: 88,
          impact: 'medium',
          category: 'Custom Focus',
          details: focusResponse,
          dataSource: fileName,
          webVerified: config.useWebVerification
        });
      }

      // Custom Prompt Analysis
      if (config.customPrompt && config.customPrompt.trim()) {
        const customResponse = await callGeminiAPI(
          `${config.customPrompt}\n\nContext: This analysis is for L'Oréal's beauty division. ${config.useWebVerification ? 'Use web search to enhance your analysis with current market information.' : ''}\n\nFormat your response with clear headers using ## for main sections and * for bullet points.`, 
          fileData, 
          config.useWebVerification
        );
        
        insights.push({
          id: `custom-${Date.now()}`,
          type: 'pattern',
          title: 'Custom Analysis Request',
          description: 'Analysis based on your specific requirements',
          confidence: 86,
          impact: 'medium',
          category: 'Custom Analysis',
          details: customResponse,
          dataSource: fileName,
          webVerified: config.useWebVerification
        });
      }

    } catch (error) {
      console.error('Gemini API Error:', error);
      throw error;
    }

    return insights;
  };

  // Auto-run analysis when config changes (Intelligence page)
  const runAnalysis = useCallback(async () => {
    if (!selectedFile) {
      setError('No data source available');
      return;
    }

    const file = jsonFiles.find(f => f.id === selectedFile);
    if (!file) {
      setError(`Data source not found`);
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    
    try {
      const newInsights = await analyzeWithLorealPrompt(file.data, analysisConfig, file.name);
      setInsights(newInsights);
      
      setJsonFiles(prev => prev.map(f => 
        f.id === selectedFile ? { ...f, lastAnalyzed: new Date() } : f
      ));
      
      if (newInsights.length > 0) {
        setExpandedInsight(newInsights[0].id);
      }
      
    } catch (error) {
      console.error('Analysis failed:', error);
      setError(error instanceof Error ? error.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile, jsonFiles, analysisConfig]);

  // Clear all data function that clears localStorage
  const clearAllData = () => {
    Object.values(STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key);
    });
    setInsights([]);
    setSelectedFile(null);
    setExpandedInsight(null);
    setAnalysisConfig(defaultConfig);
    setSearchQuery('');
    setFilterType('all');
    setShowConfig(false);
    setError(null);
  };

  const filteredInsights = insights.filter(insight => {
    const matchesSearch = searchQuery === '' || 
      insight.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      insight.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      insight.details?.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesFilter = filterType === 'all' || insight.type === filterType;
    
    return matchesSearch && matchesFilter;
  });

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'trend': return <TrendingUp className="w-4 h-4" />;
      case 'pattern': return <BarChart3 className="w-4 h-4" />;
      case 'anomaly': return <AlertTriangle className="w-4 h-4" />;
      case 'opportunity': return <Lightbulb className="w-4 h-4" />;
      case 'risk': return <AlertTriangle className="w-4 h-4" />;
      case 'summary': return <Info className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'trend': return 'text-blue-500';
      case 'pattern': return 'text-green-500';
      case 'anomaly': return 'text-yellow-500';
      case 'opportunity': return 'text-purple-500';
      case 'risk': return 'text-red-500';
      case 'summary': return 'text-indigo-500';
      default: return 'text-muted-foreground';
    }
  };

  const getImpactBadgeColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-200';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const exportInsights = () => {
    const exportData = {
      analysisDate: new Date().toISOString(),
      analysisType: 'L\'Oréal Beauty Trend Intelligence',
      webVerification: analysisConfig.useWebVerification,
      dataSource: jsonFiles.find(f => f.id === selectedFile)?.name || 'Unknown',
      config: analysisConfig,
      insights: insights
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = `loreal-beauty-intelligence-${new Date().toISOString().split('T')[0]}.json`;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const currentFile = jsonFiles.find(f => f.id === selectedFile);

  // Configuration Panel Component (Updated with fixed dark mode)
const ConfigPanel = () => (
  <div className="h-full">
    {/* Configuration Header */}
    <Alert className="border-purple-200 bg-gradient-to-r from-purple-50 to-pink-50 dark:border-purple-700 dark:from-purple-900/30 dark:to-pink-900/30 mb-6">
      <Settings className="h-5 w-5 text-purple-600 dark:text-purple-400" />
      <AlertDescription className="text-purple-800 dark:text-purple-200">
        <strong>Analysis Configuration</strong> - Configure your beauty trend analysis settings. Changes are automatically saved.
      </AlertDescription>
    </Alert>

    {/* Analysis Configuration - Fixed Background */}
    <div className="rounded-lg border border-border shadow-sm h-[calc(100vh-12rem)] bg-card">
      <div className="p-6 pb-0">
        <h3 className="text-lg font-semibold flex items-center mb-6 text-foreground">
          <Settings className="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400" />
          Analysis Settings
        </h3>
      </div>
      <div className="px-6 pb-6 space-y-6 h-full overflow-y-auto">
        <div className="grid grid-cols-1 gap-6">
          <div>
            <label className="text-sm font-medium text-foreground mb-3 block">
              Analysis Depth
            </label>
            <Select
              value={analysisConfig.analysisDepth}
              onValueChange={(value: 'quick' | 'standard' | 'deep') =>
                setAnalysisConfig(prev => ({ ...prev, analysisDepth: value }))
              }
            >
              <SelectTrigger className="h-12 border-border bg-background text-foreground hover:bg-accent hover:text-accent-foreground">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-background border-border">
                <SelectItem value="quick" className="text-foreground hover:bg-accent hover:text-accent-foreground">Quick Analysis</SelectItem>
                <SelectItem value="standard" className="text-foreground hover:bg-accent hover:text-accent-foreground">Standard Analysis</SelectItem>
                <SelectItem value="deep" className="text-foreground hover:bg-accent hover:text-accent-foreground">Deep Analysis</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-foreground mb-3 block">
              Beauty Focus Areas
            </label>
            <Input
              placeholder="skincare, makeup trends, Gen Z preferences, regional markets..."
              value={analysisConfig.focusAreas.join(', ')}
              onChange={(e) =>
                setAnalysisConfig(prev => ({
                  ...prev,
                  focusAreas: e.target.value.split(',').map(area => area.trim()).filter(Boolean)
                }))
              }
              className="h-12 border-border bg-background text-foreground placeholder:text-muted-foreground focus:border-purple-500 focus:ring-purple-500"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Specific beauty categories or market segments to focus on
            </p>
          </div>

          <div className="flex items-center space-x-3 p-4 bg-gradient-to-r from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg border border-purple-200 dark:border-purple-700">
            <input
              type="checkbox"
              checked={analysisConfig.includeRecommendations}
              onChange={(e) =>
                setAnalysisConfig(prev => ({ ...prev, includeRecommendations: e.target.checked }))
              }
              className="rounded border-border w-4 h-4 text-purple-600 focus:ring-purple-500 bg-background"
            />
            <label className="text-sm font-medium text-foreground">Include strategic recommendations & action plans</label>
          </div>
        </div>

        {/* Custom Prompt - Expanded */}
        <div className="flex-1">
          <label className="text-sm font-medium text-foreground mb-3 block">
            Additional Analysis Requirements
          </label>
          <Textarea
            placeholder="Specific questions about beauty trends, market timing, competitive analysis, or regional considerations..."
            value={analysisConfig.customPrompt || ''}
            onChange={(e) =>
              setAnalysisConfig(prev => ({ ...prev, customPrompt: e.target.value }))
            }
            className="min-h-32 resize-none border-border bg-background text-foreground placeholder:text-muted-foreground focus:border-purple-500 focus:ring-purple-500"
          />
        </div>

        {/* Save Notice - Bottom */}
        <Alert className="!bg-green-50 dark:!bg-green-900/30 border-green-200 dark:border-green-700">
          <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
          <AlertDescription className="text-green-800 dark:text-green-200">
            Configuration settings are automatically saved and will be applied when generating intelligence.
          </AlertDescription>
        </Alert>
      </div>
    </div>
  </div>
);

  // Main Analysis Display Component (Updated with fixed dark mode)
  const MainAnalysisDisplay = () => (
    <div className="space-y-6">
      {/* Intelligence Header with Controls */}
      <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 rounded-lg border border-purple-200 dark:border-purple-700">
        <div className="flex items-center space-x-4">
          <Target className="h-6 w-6 text-purple-600 dark:text-purple-400" />
          <div>
            <h2 className="text-lg font-semibold text-purple-800 dark:text-purple-200">
              L'Oréal Beauty Trend Intelligence
            </h2>
            <p className="text-sm text-purple-600 dark:text-purple-300">
              AI-powered beauty industry analysis with trend identification and strategic insights
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Data Source Display */}
          <div className="flex items-center space-x-2 px-3 py-2 bg-white/80 dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-600">
            <Database className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium text-foreground">
              {currentFile?.name || 'Forecast Data'}
            </span>
          </div>


          {/* Web Verification Toggle - Fixed */}
          <div className="flex items-center space-x-3 px-3 py-2 bg-white/80 dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-600">
          <Globe className="w-4 h-4 text-blue-600 dark:text-blue-400" />
          <span className="text-sm text-foreground">Web Verification</span>
          <input
            type="checkbox"
            checked={analysisConfig.useWebVerification}
            onChange={(e) => setAnalysisConfig(prev => ({ ...prev, useWebVerification: e.target.checked }))}
            className="w-4 h-4 text-blue-600 dark:text-blue-400 bg-background border-border rounded focus:ring-blue-500 dark:focus:ring-blue-400 focus:ring-2"
          />
          </div>

          {/* Generate Button - Fixed */}
          {/* Generate Button - Minimal Hover Change */}
        <Button 
          variant="outline" 
          size="sm" 
          onClick={runAnalysis}
          disabled={!selectedFile || isAnalyzing}
          className="border-border text-foreground hover:bg-accent hover:text-accent-foreground disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Zap className="w-4 h-4 mr-2" />
          )}
          {isAnalyzing ? "Generating..." : "Generate Intelligence"}
        </Button>
        </div>
      </div>

      {/* Quick Stats */}
      {insights.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="text-center bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/50 dark:to-purple-800/50 border-purple-200 dark:border-purple-700">
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{insights.length}</div>
              <div className="text-sm text-purple-700 dark:text-purple-300">Intelligence Reports</div>
            </CardContent>
          </Card>
          <Card className="text-center bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/50 dark:to-green-800/50 border-green-200 dark:border-green-700">
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {insights.filter(i => i.impact === 'high').length}
              </div>
              <div className="text-sm text-green-700 dark:text-green-300">High Impact Insights</div>
            </CardContent>
          </Card>
          <Card className="text-center bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/50 dark:to-blue-800/50 border-blue-200 dark:border-blue-700">
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {Math.round(insights.reduce((acc, i) => acc + i.confidence, 0) / insights.length)}%
              </div>
              <div className="text-sm text-blue-700 dark:text-blue-300">Avg Confidence</div>
            </CardContent>
          </Card>
          <Card className="text-center bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/50 dark:to-orange-800/50 border-orange-200 dark:border-orange-700">
            <CardContent className="pt-4">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {insights.filter(i => i.webVerified).length}
              </div>
              <div className="text-sm text-orange-700 dark:text-orange-300">Web Verified</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Insights Display */}
      {insights.length > 0 ? (
        <Card className="bg-card border-border">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center text-foreground">
                <Brain className="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400" />
                Beauty Intelligence Reports ({insights.length})
              </CardTitle>
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={exportInsights}
                  className="border-border text-foreground hover:bg-accent hover:text-accent-foreground"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export Intelligence
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={clearAllData}
                  className="border-border text-foreground hover:bg-accent hover:text-accent-foreground"
                >
                  Clear All
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Search and Filter */}
            <div className="flex items-center space-x-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search beauty intelligence reports..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 border-border bg-background text-foreground placeholder:text-muted-foreground"
                />
              </div>
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-40 border-border bg-background text-foreground">
                  <SelectValue placeholder="Filter by type" />
                </SelectTrigger>
                <SelectContent className="bg-background border-border">
                  <SelectItem value="all" className="text-foreground hover:bg-accent">All Types</SelectItem>
                  <SelectItem value="summary" className="text-foreground hover:bg-accent">Summary</SelectItem>
                  <SelectItem value="opportunity" className="text-foreground hover:bg-accent">Opportunities</SelectItem>
                  <SelectItem value="risk" className="text-foreground hover:bg-accent">Competitive</SelectItem>
                  <SelectItem value="pattern" className="text-foreground hover:bg-accent">Focus Areas</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Insights List */}
            <div className="space-y-4">
              {filteredInsights.map((insight) => (
                <Card key={insight.id} className="border-l-4 border-l-purple-500 bg-gradient-to-r from-card to-purple-50/30 dark:to-purple-900/20">
                  <CardHeader className="pb-3">
                    <div 
                      className="flex items-center justify-between cursor-pointer"
                      onClick={() => setExpandedInsight(
                        expandedInsight === insight.id ? null : insight.id
                      )}
                    >
                      <div className="flex items-center space-x-3">
                        <span className={getInsightColor(insight.type)}>
                          {getInsightIcon(insight.type)}
                        </span>
                        <div>
                          <h4 className="font-semibold text-lg text-foreground">{insight.title}</h4>
                          <p className="text-sm text-muted-foreground">
                            {insight.description}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge className={getImpactBadgeColor(insight.impact)}>
                          {insight.impact} impact
                        </Badge>
                        <Badge variant="outline" className="text-xs border-border">
                          {insight.confidence}% confidence
                        </Badge>
                        {insight.webVerified && (
                          <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-200">
                            <Globe className="w-3 h-3 mr-1" />
                            Web Verified
                          </Badge>
                        )}
                        {expandedInsight === insight.id ? 
                          <ChevronUp className="w-4 h-4 text-muted-foreground" /> : 
                          <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        }
                      </div>
                    </div>
                  </CardHeader>
                  
                  {expandedInsight === insight.id && (
                    <CardContent className="pt-0">
                      <div className="bg-gradient-to-r from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-700">
                        <div className="flex items-center justify-between mb-3">
                          <Badge variant="outline" className="bg-card border-border">
                            {insight.category}
                          </Badge>
                          <div className="flex items-center space-x-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => navigator.clipboard.writeText(insight.details || insight.description)}
                              className="text-foreground hover:bg-accent hover:text-accent-foreground"
                            >
                              <Copy className="w-4 h-4 mr-2" />
                              Copy Report
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                const shareData = {
                                  title: insight.title,
                                  text: insight.description,
                                  url: window.location.href
                                };
                                if (navigator.share) {
                                  navigator.share(shareData);
                                } else {
                                  navigator.clipboard.writeText(`${insight.title}\n${insight.description}\n${window.location.href}`);
                                }
                              }}
                              className="text-foreground hover:bg-accent hover:text-accent-foreground"
                            >
                              <Share2 className="w-4 h-4 mr-2" />
                              Share
                            </Button>
                          </div>
                        </div>
                        {insight.details && (
                          <div className="prose prose-sm max-w-none dark:prose-invert">
                            <div className="text-foreground max-h-96 overflow-y-auto">
                              {parseMarkdownToJSX(insight.details)}
                            </div>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  )}
                </Card>
              ))}
            </div>

            {filteredInsights.length === 0 && insights.length > 0 && (
              <div className="text-center py-8">
                <Search className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
                <p className="text-muted-foreground">No reports match your search criteria</p>
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <Card className="bg-card border-border">
          <CardContent className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 rounded-full flex items-center justify-center">
              <Target className="w-8 h-8 text-purple-600 dark:text-purple-400" />
            </div>
            <p className="text-lg font-medium text-foreground mb-2">Ready to Generate Beauty Intelligence</p>
            <p className="text-muted-foreground mb-4">
              Click "Generate Intelligence" above to create comprehensive beauty trend analysis
            </p>
            {!currentFile && (
              <Alert className="border-yellow-200 bg-yellow-50 dark:border-yellow-700 dark:bg-yellow-900/30 max-w-md mx-auto">
                <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                <AlertDescription className="text-yellow-800 dark:text-yellow-200">
                  Loading forecast data...
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );

  return (
    <div className={`p-6 space-y-6 bg-background ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            L'Oréal Beauty Trend Intelligence
          </h1>
          <p className="text-muted-foreground mt-1">
            AI-powered beauty industry analysis with trend identification, generational insights, and strategic timing
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {/* Toggle Switch - Fixed with proper dark mode support */}
<div className="flex items-center space-x-2">
  <Button
    onClick={() => setShowConfig(false)}
    variant={!showConfig ? "default" : "outline"}
    size="sm"
    className={!showConfig 
      ? "bg-gradient-to-r from-purple-600 to-pink-600 text-black dark:text-white font-medium border border-transparent hover:border-purple-300/50 dark:hover:border-purple-400/50 shadow-sm hover:shadow-lg hover:scale-[1.02] transition-all duration-200"
      : "border-border text-foreground hover:bg-accent hover:text-accent-foreground hover:scale-[1.02] transition-all duration-200"
    }
  >
    <Brain className="w-4 h-4 mr-2" />
    Intelligence
  </Button>
  
  <Button
    onClick={() => setShowConfig(true)}
    variant={showConfig ? "default" : "outline"}
    size="sm"
    className={showConfig 
      ? "bg-gradient-to-r from-purple-600 to-pink-600 text-black dark:text-white font-medium border border-transparent hover:border-purple-300/50 dark:hover:border-purple-400/50 shadow-sm hover:shadow-lg hover:scale-[1.02] transition-all duration-200"
      : "border-border text-foreground hover:bg-accent hover:text-accent-foreground hover:scale-[1.02] transition-all duration-200"
    }
  >
    <Settings className="w-4 h-4 mr-2" />
    Configuration
  </Button>
</div>

          <Button 
            variant="outline" 
            size="sm" 
            onClick={loadJsonFiles} 
            disabled={isLoadingFiles}
            className="border-purple-300 dark:border-purple-600 text-purple-700 dark:text-purple-300 hover:bg-purple-50 dark:hover:bg-purple-900/30 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoadingFiles ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4 mr-2" />
            )}
            Refresh Data
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert className="border-red-200 bg-red-50 dark:border-red-700 dark:bg-red-900/30">
          <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
          <AlertDescription className="text-red-800 dark:text-red-200">
            {error}
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setError(null)}
              className="ml-2 border-red-200 dark:border-red-600 text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/30"
            >
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Content - Toggle between Config and Analysis */}
      {showConfig ? <ConfigPanel /> : <MainAnalysisDisplay />}

      {/* Loading Overlay */}
      {isAnalyzing && (
        <div className="fixed inset-0 bg-black/20 dark:bg-black/50 flex items-center justify-center z-50">
          <Card className="p-8 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/50 dark:to-pink-900/50 border border-purple-200 dark:border-purple-700 shadow-xl">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Loader2 className="w-10 h-10 animate-spin text-purple-600 dark:text-purple-400" />
                <div className="absolute inset-0 w-10 h-10 animate-pulse">
                  <Target className="w-10 h-10 text-pink-400 dark:text-pink-300 opacity-50" />
                </div>
              </div>
              <div>
                <p className="font-medium text-lg text-purple-800 dark:text-purple-200">
                  Generating Beauty Intelligence...
                </p>
                <p className="text-sm text-purple-600 dark:text-purple-300">
                  {analysisConfig.useWebVerification 
                    ? 'Analyzing trends and verifying with real-time web data...'
                    : 'Processing beauty trend forecast data...'
                  }
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

export default AIInsightsDashboard;

