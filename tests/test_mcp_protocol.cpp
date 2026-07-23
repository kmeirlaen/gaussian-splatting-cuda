/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>

#include "config.h"
#include "mcp/mcp_http_server.hpp"
#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/mcp_tools.hpp"

#include <httplib/httplib.h>

#include <cstdint>
#include <stdexcept>

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

    TEST(McpProtocolTest, ParseRequestExtractsIdForEachIdKind) {
        const auto int_req = parse_request(R"({"jsonrpc":"2.0","id":42,"method":"ping"})");
        EXPECT_EQ(int_req.id, RequestId(int64_t{42}));

        const auto string_req = parse_request(R"({"jsonrpc":"2.0","id":"abc-123","method":"ping"})");
        EXPECT_EQ(string_req.id, RequestId(std::string("abc-123")));

        const auto null_req = parse_request(R"({"jsonrpc":"2.0","id":null,"method":"ping"})");
        EXPECT_EQ(null_req.id, RequestId(nullptr));

        const auto notification_req = parse_request(R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
        EXPECT_EQ(notification_req.id, RequestId());

        EXPECT_THROW(parse_request("{not json"), json::parse_error);
    }

    TEST(McpProtocolTest, RequestIdIntegerEchoedOnSuccessAndErrorPaths) {
        McpServer server;
        const auto success = server.handle_request(JsonRpcRequest{.id = int64_t{7}, .method = "ping"});
        EXPECT_EQ(success.id, RequestId(int64_t{7}));
        EXPECT_EQ(json::parse(serialize_response(success))["id"], 7);

        const auto failure = server.handle_request(JsonRpcRequest{.id = int64_t{7}, .method = "unknown/method"});
        EXPECT_EQ(failure.id, RequestId(int64_t{7}));
        EXPECT_EQ(json::parse(serialize_response(failure))["id"], 7);
    }

    TEST(McpProtocolTest, RequestIdStringEchoedOnSuccessAndErrorPaths) {
        McpServer server;
        const auto success = server.handle_request(JsonRpcRequest{.id = std::string("req-a"), .method = "ping"});
        EXPECT_EQ(success.id, RequestId(std::string("req-a")));
        EXPECT_EQ(json::parse(serialize_response(success))["id"], "req-a");

        const auto failure = server.handle_request(JsonRpcRequest{.id = std::string("req-a"), .method = "unknown/method"});
        EXPECT_EQ(failure.id, RequestId(std::string("req-a")));
        EXPECT_EQ(json::parse(serialize_response(failure))["id"], "req-a");
    }

    TEST(McpProtocolTest, RequestIdNullSerializesAsJsonNull) {
        McpServer server;
        const auto response = server.handle_request(JsonRpcRequest{.id = nullptr, .method = "unknown/method"});
        EXPECT_EQ(response.id, RequestId(nullptr));

        const auto body = json::parse(serialize_response(response));
        ASSERT_TRUE(body.contains("id"));
        EXPECT_TRUE(body["id"].is_null());
    }

    TEST(McpProtocolTest, RequestIdAbsentOmitsIdFieldOnSerialization) {
        McpServer server;
        JsonRpcRequest req;
        req.method = "ping";
        EXPECT_EQ(req.id, RequestId());

        const auto response = server.handle_request(req);
        EXPECT_EQ(response.id, RequestId());

        const auto body = json::parse(serialize_response(response));
        EXPECT_FALSE(body.contains("id"));
    }

    TEST(McpProtocolTest, WireSerializationIsTotalOverIllFormedUtf8) {
        JsonRpcResponse response;
        response.id = 7;
        response.result = json{{"error", std::string("bad \xC3 byte")}};
        std::string serialized;
        EXPECT_NO_THROW(serialized = serialize_response(response));
        EXPECT_NE(serialized.find("\xEF\xBF\xBD"), std::string::npos);

        EXPECT_NO_THROW(
            serialized = serialize_notification("event", json{{"error", std::string("\xFF")}}));
        EXPECT_NE(serialized.find("\xEF\xBF\xBD"), std::string::npos);
    }

    TEST(McpProtocolTest, ToolsCallMissingNameReturnsInvalidParams) {
        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{.id = int64_t{1}, .method = "initialize"}).result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{{"arguments", json::object()}}});

        ASSERT_TRUE(response.error.has_value());
        EXPECT_EQ(response.error->code, JsonRpcError::INVALID_PARAMS);
        EXPECT_EQ(response.id, RequestId(int64_t{2}));
    }

    TEST(McpProtocolTest, ToolsCallNonStringNameReturnsInvalidParamsNotInternalError) {
        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{.id = int64_t{1}, .method = "initialize"}).result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{{"name", 123}, {"arguments", json::object()}}});

        ASSERT_TRUE(response.error.has_value());
        EXPECT_EQ(response.error->code, JsonRpcError::INVALID_PARAMS);
        EXPECT_EQ(response.id, RequestId(int64_t{2}));
    }

    TEST(McpHttpServerTest, MalformedJsonBodyRespondsWithNullIdAndParseError) {
        McpHttpServer server;
        ASSERT_TRUE(server.start(47691));

        httplib::Client client("127.0.0.1", 47691);
        const auto res = client.Post("/mcp", "{not json", "application/json");
        ASSERT_TRUE(res);

        const auto body = json::parse(res->body);
        ASSERT_TRUE(body.contains("id"));
        EXPECT_TRUE(body["id"].is_null());
        ASSERT_TRUE(body.contains("error"));
        EXPECT_EQ(body["error"]["code"], JsonRpcError::PARSE_ERROR);

        server.stop();
    }

    TEST(McpHttpServerTest, ToolHandlerThrowRespondsWithInternalErrorAndEchoedIdNoLeak) {
        static constexpr const char* tool_name = "test.throwing_tool";
        static constexpr const char* leaked_detail = "sensitive internal detail";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Throws for firewall testing",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "command"}},
            [](const json&) -> json {
                throw std::runtime_error(leaked_detail);
            });

        McpHttpServer server;
        ASSERT_TRUE(server.start(47692));

        httplib::Client client("127.0.0.1", 47692);

        const json init_req{
            {"jsonrpc", "2.0"},
            {"id", 1},
            {"method", "initialize"},
            {"params", json::object()}};
        ASSERT_TRUE(client.Post("/mcp", init_req.dump(), "application/json"));

        const json call_req{
            {"jsonrpc", "2.0"},
            {"id", "req-42"},
            {"method", "tools/call"},
            {"params", json{{"name", tool_name}, {"arguments", json::object()}}}};
        const auto res = client.Post("/mcp", call_req.dump(), "application/json");
        ASSERT_TRUE(res);

        // A thrown tool handler is now a tool-execution error, not a protocol
        // error: the registry catches it and returns a successful JSON-RPC
        // response whose tool result carries a stable envelope. The two
        // load-bearing invariants are preserved: the id is echoed and the
        // thrown detail appears nowhere in the body.
        EXPECT_EQ(res->body.find(leaked_detail), std::string::npos);

        const auto body = json::parse(res->body);
        ASSERT_TRUE(body.contains("id"));
        EXPECT_EQ(body["id"], "req-42");
        ASSERT_TRUE(body.contains("result"));
        const auto& result = body["result"];
        EXPECT_TRUE(result["isError"].get<bool>());
        ASSERT_TRUE(result["structuredContent"].contains("error"));
        EXPECT_EQ(result["structuredContent"]["error"]["code"], "Internal");

        server.stop();
    }

    TEST(McpHttpServerTest, HandlerThrowOutsideToolBecomesInternalErrorWithEnvelopeData) {
        static constexpr const char* prefix = "lichtfeld://throwtest/";
        static constexpr const char* leaked_detail = "resource sensitive detail";
        ScopedResourcePrefixRegistration cleanup(prefix);

        ResourceRegistry::instance().register_resource_prefix(
            prefix,
            [](const std::string&) -> std::expected<std::vector<McpResourceContent>, std::string> {
                throw std::runtime_error(leaked_detail);
            });

        McpHttpServer server;
        ASSERT_TRUE(server.start(47693));

        httplib::Client client("127.0.0.1", 47693);

        const json init_req{
            {"jsonrpc", "2.0"},
            {"id", 1},
            {"method", "initialize"},
            {"params", json::object()}};
        ASSERT_TRUE(client.Post("/mcp", init_req.dump(), "application/json"));

        const json read_req{
            {"jsonrpc", "2.0"},
            {"id", "res-7"},
            {"method", "resources/read"},
            {"params", json{{"uri", "lichtfeld://throwtest/item"}}}};
        const auto res = client.Post("/mcp", read_req.dump(), "application/json");
        ASSERT_TRUE(res);

        EXPECT_EQ(res->body.find(leaked_detail), std::string::npos);

        const auto body = json::parse(res->body);
        EXPECT_EQ(body["id"], "res-7");
        ASSERT_TRUE(body.contains("error"));
        EXPECT_EQ(body["error"]["code"], JsonRpcError::INTERNAL_ERROR);
        ASSERT_TRUE(body["error"].contains("data"));
        EXPECT_TRUE(body["error"]["data"].contains("code"));
        EXPECT_EQ(body["error"]["data"]["domain"], "MCP");

        server.stop();
    }

    TEST(McpProtocolTest, LegacyStringErrorBecomesEnvelopeWithCompatMirror) {
        static constexpr const char* tool_name = "test.legacy_string_error";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Legacy string error handler",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json {
                return json{{"error", "No scene loaded"}, {"detail_field", 7}};
            });

        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{
                                              .id = int64_t{1},
                                              .method = "initialize",
                                              .params = json::object()})
                        .result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{{"name", tool_name}, {"arguments", json::object()}}});

        ASSERT_TRUE(response.result.has_value());
        const auto& structured = (*response.result)["structuredContent"];
        EXPECT_TRUE((*response.result)["isError"].get<bool>());
        EXPECT_EQ(structured["error"]["code"], "FailedPrecondition");
        EXPECT_EQ(structured["error"]["domain"], "MCP");
        EXPECT_EQ(structured["error"]["message"], "No scene loaded");
        EXPECT_FALSE(structured["error"]["retryable"].get<bool>());
        EXPECT_EQ(structured["error_message"], "No scene loaded");
        EXPECT_EQ(structured["detail_field"], 7);
    }

    TEST(McpProtocolTest, ToolNotFoundAndMissingParameterYieldTypedEnvelopes) {
        const auto not_found =
            ToolRegistry::instance().call_tool("does.not.exist", json::object());
        ASSERT_TRUE(not_found.contains("error"));
        EXPECT_EQ(not_found["error"]["code"], "NotFound");
        EXPECT_EQ(not_found["error"]["details"]["parameter"], "does.not.exist");
        EXPECT_EQ(not_found["error_message"], "Tool not found: does.not.exist");

        static constexpr const char* tool_name = "test.requires_param";
        ScopedToolRegistration cleanup(tool_name);
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Requires a parameter",
                .input_schema = {.type = "object", .properties = json::object(), .required = {"value"}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json { return json{{"success", true}}; });

        const auto missing = ToolRegistry::instance().call_tool(tool_name, json::object());
        ASSERT_TRUE(missing.contains("error"));
        EXPECT_EQ(missing["error"]["code"], "InvalidArgument");
        EXPECT_EQ(missing["error"]["details"]["parameter"], "value");
        EXPECT_EQ(missing["error_message"], "Missing required parameter: value");
    }

    TEST(McpProtocolTest, TypedEnvelopeHandlerResultIsPassedThroughWithMirror) {
        static constexpr const char* tool_name = "test.typed_envelope";
        ScopedToolRegistration cleanup(tool_name);
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Emits a typed envelope directly",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json {
                return json{{"error", json{
                                          {"code", "NotFound"},
                                          {"domain", "IO"},
                                          {"message", "Dataset was not found"},
                                          {"retryable", false},
                                          {"operation_id", 0}}}};
            });

        const auto result = ToolRegistry::instance().call_tool(tool_name, json::object());
        ASSERT_TRUE(result["error"].is_object());
        EXPECT_EQ(result["error"]["code"], "NotFound");
        EXPECT_EQ(result["error"]["domain"], "IO");
        EXPECT_EQ(result["error_message"], "Dataset was not found");
    }

    TEST(McpProtocolTest, ResourceUnknownUriYieldsNotFoundEnvelopeData) {
        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{
                                              .id = int64_t{1},
                                              .method = "initialize",
                                              .params = json::object()})
                        .result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "resources/read",
            .params = json{{"uri", "lichtfeld://nonexistent/thing"}}});

        ASSERT_TRUE(response.error.has_value());
        EXPECT_EQ(response.error->code, JsonRpcError::INVALID_PARAMS);
        ASSERT_TRUE(response.error->data.has_value());
        EXPECT_EQ((*response.error->data)["code"], "NotFound");
        EXPECT_EQ((*response.error->data)["domain"], "MCP");
    }

    TEST(McpProtocolTest, RegistryCaughtHandlerFailureCarriesOperationId) {
        static constexpr const char* tool_name = "test.correlated_throw";
        ScopedToolRegistration cleanup(tool_name);
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Throws for correlation testing",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "command"}},
            [](const json&) -> json { throw std::runtime_error("boom"); });

        const lfs::OperationId operation_id = lfs::OperationId::generate();
        const auto result =
            ToolRegistry::instance().call_tool(tool_name, json::object(), operation_id);

        ASSERT_TRUE(result.contains("error"));
        EXPECT_NE(operation_id.value(), 0u);
        EXPECT_EQ(result["error"]["operation_id"].get<std::uint64_t>(), operation_id.value());
        EXPECT_EQ(result["error"]["code"], "Internal");
    }

} // namespace lfs::mcp
