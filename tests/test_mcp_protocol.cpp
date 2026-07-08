/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>

#include "config.h"
#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/mcp_tools.hpp"

namespace lfs::mcp {

    namespace {

        class ScopedToolRegistration {
        public:
            explicit ScopedToolRegistration(std::string name) : name_(std::move(name)) {}
            ~ScopedToolRegistration() {
                ToolRegistry::instance().unregister_tool(name_);
            }

        private:
            std::string name_;
        };

        class ScopedResourcePrefixRegistration {
        public:
            explicit ScopedResourcePrefixRegistration(std::string prefix) : prefix_(std::move(prefix)) {}
            ~ScopedResourcePrefixRegistration() {
                ResourceRegistry::instance().unregister_resource_prefix(prefix_);
            }

        private:
            std::string prefix_;
        };

        bool is_claude_compatible_tool_name(const std::string& name) {
            if (name.empty() || name.size() > 64)
                return false;

            for (const unsigned char ch : name) {
                if (!std::isalnum(ch) && ch != '_' && ch != '-')
                    return false;
            }
            return true;
        }

    } // namespace

    TEST(McpProtocolTest, ToolJsonKeepsStandardAnnotationsAndMovesLichtFeldMetadataToMeta) {
        const auto payload = tool_to_json(McpTool{
            .name = "test.describe",
            .description = "Describe metadata",
            .input_schema = {.type = "object", .properties = json::object(), .required = {}},
            .metadata = McpToolMetadata{
                .category = "editor",
                .kind = "query",
                .runtime = "gui",
                .thread_affinity = "gui_thread",
                .destructive = false,
                .long_running = true,
            }});

        ASSERT_TRUE(payload.contains("annotations"));
        const auto& annotations = payload["annotations"];
        EXPECT_TRUE(annotations["readOnlyHint"].get<bool>());
        EXPECT_TRUE(annotations["idempotentHint"].get<bool>());
        EXPECT_FALSE(annotations["destructiveHint"].get<bool>());
        for (const auto& item : annotations.items()) {
            EXPECT_EQ(item.key().find("x-lfs-"), std::string::npos) << item.key();
        }

        ASSERT_TRUE(payload.contains("_meta"));
        const auto& meta = payload["_meta"];
        EXPECT_EQ(meta["app.lichtfeld/category"], "editor");
        EXPECT_EQ(meta["app.lichtfeld/kind"], "query");
        EXPECT_EQ(meta["app.lichtfeld/runtime"], "gui");
        EXPECT_EQ(meta["app.lichtfeld/thread_affinity"], "gui_thread");
        EXPECT_TRUE(meta["app.lichtfeld/user_visible"].get<bool>());
        EXPECT_TRUE(meta["app.lichtfeld/long_running"].get<bool>());
    }

    TEST(McpProtocolTest, ToolJsonSerializesNullSchemaPropertiesAsObject) {
        const auto payload = tool_to_json(McpTool{
            .name = "test.empty_schema",
            .description = "Empty schema properties should serialize as an object",
            .input_schema = {.type = "object", .properties = json(), .required = {}}});

        ASSERT_TRUE(payload.contains("inputSchema"));
        const auto& schema = payload["inputSchema"];
        ASSERT_TRUE(schema.contains("properties"));
        EXPECT_TRUE(schema["properties"].is_object());
        EXPECT_TRUE(schema["properties"].empty());
    }

    TEST(McpProtocolTest, InitializeReportsBuildVersion) {
        McpServer server;
        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        ASSERT_TRUE(result.contains("serverInfo"));
        EXPECT_EQ(result["serverInfo"]["name"], "lichtfeld-mcp");
        EXPECT_EQ(result["serverInfo"]["version"], GIT_TAGGED_VERSION);
        EXPECT_NE(result["serverInfo"]["version"], "1.0.0");
    }

    TEST(McpProtocolTest, ToolCallReturnsStructuredContent) {
        static constexpr const char* tool_name = "test.structured_response";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Structured response test",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json& args) -> json {
                return json{
                    {"success", true},
                    {"echo", args.value("value", 0)},
                };
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{
                {"name", tool_name},
                {"arguments", json{{"value", 42}}},
            }});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        ASSERT_TRUE(result.contains("structuredContent"));
        EXPECT_EQ(result["structuredContent"]["echo"], 42);
        EXPECT_FALSE(result["isError"].get<bool>());
        ASSERT_TRUE(result.contains("content"));
        ASSERT_TRUE(result["content"].is_array());
        ASSERT_FALSE(result["content"].empty());
        EXPECT_NE(result["content"][0]["text"].get<std::string>().find("\"echo\": 42"), std::string::npos);
    }

    TEST(McpProtocolTest, ToolCallIgnoresEmptyErrorStringForTransportErrors) {
        static constexpr const char* tool_name = "test.empty_error_string";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Empty error string should not mark transport failure",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json {
                return json{
                    {"success", true},
                    {"error", ""},
                };
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{
                {"name", tool_name},
                {"arguments", json::object()},
            }});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        EXPECT_FALSE(result["isError"].get<bool>());
        EXPECT_EQ(result["structuredContent"]["error"], "");
    }

    TEST(McpProtocolTest, ToolsListSerializesNullSchemaPropertiesAsObjectAndCallContractIsUnchanged) {
        static constexpr const char* tool_name = "test.null_schema_properties";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Null schema properties test",
                .input_schema = {.type = "object", .properties = json(), .required = {}}},
            [](const json&) -> json {
                return json{
                    {"success", true},
                    {"state", "ok"},
                };
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto list_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/list",
            .params = json::object()});
        ASSERT_TRUE(list_response.result.has_value());

        const auto& tools = (*list_response.result)["tools"];
        const auto found = std::find_if(tools.begin(), tools.end(), [](const json& tool) {
            return tool.value("name", "") == "test_null_schema_properties";
        });
        ASSERT_NE(found, tools.end());
        ASSERT_TRUE((*found)["inputSchema"].contains("properties"));
        EXPECT_TRUE((*found)["inputSchema"]["properties"].is_object());
        EXPECT_TRUE((*found)["inputSchema"]["properties"].empty());

        const auto call_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{3},
            .method = "tools/call",
            .params = json{
                {"name", tool_name},
                {"arguments", json::object()},
            }});
        ASSERT_TRUE(call_response.result.has_value());
        const auto& result = *call_response.result;
        ASSERT_TRUE(result.contains("content"));
        ASSERT_TRUE(result.contains("structuredContent"));
        ASSERT_TRUE(result.contains("isError"));
        EXPECT_FALSE(result["isError"].get<bool>());
        EXPECT_EQ(result["structuredContent"]["state"], "ok");
    }

    TEST(McpProtocolTest, ToolNamesAreNormalizedForListAndCallsAcceptBothForms) {
        static constexpr const char* dotted_tool_name = "test.normalized_tool";
        static constexpr const char* normalized_tool_name = "test_normalized_tool";
        ScopedToolRegistration cleanup(dotted_tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = dotted_tool_name,
                .description = "Normalized tool name test",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json& args) -> json {
                return json{
                    {"success", true},
                    {"marker", "normalized"},
                    {"value", args.value("value", 0)},
                };
            });

        const auto registered_tools = ToolRegistry::instance().list_tools();
        const auto internal_tool = std::find_if(
            registered_tools.begin(),
            registered_tools.end(),
            [](const McpTool& tool) { return tool.name == dotted_tool_name; });
        ASSERT_NE(internal_tool, registered_tools.end());

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto list_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/list",
            .params = json::object()});
        ASSERT_TRUE(list_response.result.has_value());

        bool found_normalized_name = false;
        for (const auto& tool : (*list_response.result)["tools"]) {
            const auto name = tool["name"].get<std::string>();
            EXPECT_TRUE(is_claude_compatible_tool_name(name)) << name;
            EXPECT_NE(name, dotted_tool_name);
            if (name == normalized_tool_name)
                found_normalized_name = true;
        }
        EXPECT_TRUE(found_normalized_name);

        const auto dotted_call_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{3},
            .method = "tools/call",
            .params = json{
                {"name", dotted_tool_name},
                {"arguments", json{{"value", 7}}},
            }});
        ASSERT_TRUE(dotted_call_response.result.has_value());
        EXPECT_FALSE((*dotted_call_response.result)["isError"].get<bool>());
        EXPECT_EQ((*dotted_call_response.result)["structuredContent"]["marker"], "normalized");
        EXPECT_EQ((*dotted_call_response.result)["structuredContent"]["value"], 7);

        const auto normalized_call_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{4},
            .method = "tools/call",
            .params = json{
                {"name", normalized_tool_name},
                {"arguments", json{{"value", 11}}},
            }});
        ASSERT_TRUE(normalized_call_response.result.has_value());
        EXPECT_FALSE((*normalized_call_response.result)["isError"].get<bool>());
        EXPECT_EQ((*normalized_call_response.result)["structuredContent"]["marker"], "normalized");
        EXPECT_EQ((*normalized_call_response.result)["structuredContent"]["value"], 11);
        ASSERT_TRUE((*normalized_call_response.result).contains("content"));
        ASSERT_TRUE((*normalized_call_response.result).contains("structuredContent"));
    }

    TEST(McpProtocolTest, ResourceReadUsesMostSpecificPrefixHandler) {
        static constexpr std::string_view broad_prefix = "lichtfeld://test/";
        static constexpr std::string_view narrow_prefix = "lichtfeld://test/items/";
        ScopedResourcePrefixRegistration cleanup_broad{std::string(broad_prefix)};
        ScopedResourcePrefixRegistration cleanup_narrow{std::string(narrow_prefix)};

        ResourceRegistry::instance().register_resource_prefix(
            std::string(broad_prefix),
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = json{{"handler", "broad"}}.dump()}};
            });

        ResourceRegistry::instance().register_resource_prefix(
            std::string(narrow_prefix),
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = json{
                            {"handler", "narrow"},
                            {"id", uri.substr(narrow_prefix.size())}}
                                       .dump()}};
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "resources/read",
            .params = json{{"uri", "lichtfeld://test/items/example"}}});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        ASSERT_TRUE(result.contains("contents"));
        ASSERT_TRUE(result["contents"].is_array());
        ASSERT_EQ(result["contents"].size(), 1);

        const auto parsed = json::parse(result["contents"][0]["text"].get<std::string>());
        EXPECT_EQ(parsed["handler"], "narrow");
        EXPECT_EQ(parsed["id"], "example");
    }

} // namespace lfs::mcp
