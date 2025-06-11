-- Crawl4AI Documentation Crawler Database Schema
-- This schema supports multi-project documentation crawling with vector search capabilities

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Sources table: Stores information about crawl sources
CREATE TABLE IF NOT EXISTS sources (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    project_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    base_url TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'website', 'sitemap', 'text_file'
    crawl_config JSONB DEFAULT '{}',
    last_crawled_at TIMESTAMP WITH TIME ZONE,
    total_pages INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'crawling', 'completed', 'failed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Crawled pages table: Stores individual page data
CREATE TABLE IF NOT EXISTS crawled_pages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    source_id UUID REFERENCES sources(id) ON DELETE CASCADE,
    project_id VARCHAR(50) NOT NULL,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    content TEXT,
    clean_content TEXT, -- Processed content for search
    markdown_content TEXT, -- Markdown version of content
    meta_data JSONB DEFAULT '{}',
    chunk_count INTEGER DEFAULT 0,
    embedding vector(1536), -- OpenAI embeddings dimension
    crawled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'success', -- 'success', 'failed', 'pending'
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Code examples table: Extracted code snippets from documentation
CREATE TABLE IF NOT EXISTS code_examples (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    page_id UUID REFERENCES crawled_pages(id) ON DELETE CASCADE,
    project_id VARCHAR(50) NOT NULL,
    language VARCHAR(50),
    code TEXT NOT NULL,
    description TEXT,
    context TEXT, -- Surrounding text context
    embedding vector(1536),
    position_in_page INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content chunks table: For RAG chunking strategy
CREATE TABLE IF NOT EXISTS content_chunks (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    page_id UUID REFERENCES crawled_pages(id) ON DELETE CASCADE,
    project_id VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Search history table: Track user searches for analytics
CREATE TABLE IF NOT EXISTS search_history (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    project_id VARCHAR(50) NOT NULL,
    query TEXT NOT NULL,
    query_embedding vector(1536),
    results_count INTEGER DEFAULT 0,
    search_type VARCHAR(50), -- 'semantic', 'keyword', 'hybrid'
    user_session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Project statistics table: Aggregated stats per project
CREATE TABLE IF NOT EXISTS project_statistics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    project_id VARCHAR(50) NOT NULL UNIQUE,
    total_sources INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    total_code_examples INTEGER DEFAULT 0,
    total_searches INTEGER DEFAULT 0,
    storage_used_mb INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_sources_project_id ON sources(project_id);
CREATE INDEX idx_crawled_pages_project_id ON crawled_pages(project_id);
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages(source_id);
CREATE INDEX idx_code_examples_project_id ON code_examples(project_id);
CREATE INDEX idx_code_examples_page_id ON code_examples(page_id);
CREATE INDEX idx_content_chunks_project_id ON content_chunks(project_id);
CREATE INDEX idx_content_chunks_page_id ON content_chunks(page_id);
CREATE INDEX idx_search_history_project_id ON search_history(project_id);

-- Create vector indexes for similarity search
CREATE INDEX idx_crawled_pages_embedding ON crawled_pages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_code_examples_embedding ON code_examples USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_content_chunks_embedding ON content_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to tables with updated_at
CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_crawled_pages_updated_at BEFORE UPDATE ON crawled_pages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to update project statistics
CREATE OR REPLACE FUNCTION update_project_statistics(p_project_id VARCHAR(50))
RETURNS VOID AS $$
BEGIN
    INSERT INTO project_statistics (
        project_id,
        total_sources,
        total_pages,
        total_chunks,
        total_code_examples,
        total_searches,
        storage_used_mb
    )
    VALUES (
        p_project_id,
        (SELECT COUNT(*) FROM sources WHERE project_id = p_project_id),
        (SELECT COUNT(*) FROM crawled_pages WHERE project_id = p_project_id),
        (SELECT COUNT(*) FROM content_chunks WHERE project_id = p_project_id),
        (SELECT COUNT(*) FROM code_examples WHERE project_id = p_project_id),
        (SELECT COUNT(*) FROM search_history WHERE project_id = p_project_id),
        (SELECT COALESCE(SUM(pg_column_size(content) + pg_column_size(clean_content) + pg_column_size(markdown_content)) / 1024 / 1024, 0)::INTEGER 
         FROM crawled_pages WHERE project_id = p_project_id)
    )
    ON CONFLICT (project_id) DO UPDATE SET
        total_sources = EXCLUDED.total_sources,
        total_pages = EXCLUDED.total_pages,
        total_chunks = EXCLUDED.total_chunks,
        total_code_examples = EXCLUDED.total_code_examples,
        total_searches = EXCLUDED.total_searches,
        storage_used_mb = EXCLUDED.storage_used_mb,
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;